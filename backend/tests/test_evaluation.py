import unittest
import zipfile
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from app.domain.datasources import DataPackage, FileEntry
from app.domain.extraction import (
    ExtractionContext,
    ExtractionNormalization,
    ExtractionRunResult,
    ExtractionRunState,
    FileRankingResult,
    QualitativeAttribute,
    QualitativeAttributeNormalization,
    RankedFile,
    VocabularyTermMapping,
)
from app.evaluation.models import (
    EvaluationReference,
    ReferenceAttribute,
    ReferenceObject,
    ReferenceVocabMapping,
)
from app.evaluation.scoring import evaluate_extraction_result
from app.evaluation.runner import score_reference_directory
from app.services.extraction_service import ExtractionService


def context_payload() -> dict:
    return {
        "extraction_objects": [
            {
                "object_kind": "Method",
                "extracted_object": {
                    "identifier": "ir-method",
                    "description": "Infrared spectroscopy using a Bruker ALPHA instrument.",
                    "keywords": ["IR"],
                    "has_qualitative_attributes": [
                        {"title": "method", "value": "IR"}
                    ],
                },
                "source_text": "instrument: Bruker ALPHA",
            },
            {
                "object_kind": "AgenticEntity",
                "extracted_object": {
                    "identifier": "Bruker ALPHA",
                    "description": "Bruker ALPHA spectrometer.",
                    "type": "instrument",
                },
                "source_text": "instrument: Bruker ALPHA",
            },
        ]
    }


class EvaluationTests(unittest.TestCase):
    def test_context_with_resource_inventory_adds_package_files(self):
        data_package = DataPackage(
            file_name="IR",
            files=[
                FileEntry(
                    file_path="dataset_description.txt",
                    file_name="dataset_description.txt",
                    file_extension=".txt",
                    raw_content=b"dataset name: IR",
                ),
                FileEntry(
                    file_path="SG-V4050.edit.png",
                    file_name="SG-V4050.edit.png",
                    file_extension=".png",
                    raw_content=b"image",
                ),
                FileEntry(
                    file_path="SG-V4050.infer.json",
                    file_name="SG-V4050.infer.json",
                    file_extension=".json",
                    raw_content=b"{}",
                ),
            ],
        )

        context = ExtractionService._context_with_resource_inventory(
            data_package=data_package,
            context=ExtractionContext.model_validate(context_payload()),
        )

        resources = context.resources
        self.assertIn("dataset_description.txt", [item.identifier for item in resources])
        self.assertIn("SG-V4050.infer.json", [item.identifier for item in resources])
        image = next(item for item in resources if item.identifier == "SG-V4050.edit.png")
        self.assertEqual(image.type, "image")

    def test_evaluate_extraction_result_scores_expected_items(self):
        result = ExtractionRunResult(
            document={"title": "IR"},
            extraction_context=ExtractionContext.model_validate(context_payload()),
            normalization=ExtractionNormalization(
                qualitative_attributes=[
                    QualitativeAttributeNormalization(
                        attribute=QualitativeAttribute(title="method", value="IR"),
                        term=VocabularyTermMapping(
                            source_value="IR",
                            vocabulary_identifier="http://purl.obolibrary.org/obo/chmo.owl",
                            rdf_type="owl__Class",
                            selected_uri="http://purl.obolibrary.org/obo/CHMO_0000634",
                        ),
                    )
                ]
            ),
        )
        state = ExtractionRunState(
            ranked_files=FileRankingResult(
                files=[
                    RankedFile(rank=1, file_path="dataset_description.txt"),
                    RankedFile(rank=2, file_path="SG-V4050.dx"),
                ]
            ).files
        )
        reference = EvaluationReference(
            dataset_filename="IR-IR.zip",
            relevant_files=["dataset_description.txt", "SG-V4050.dx"],
            expected_objects=[
                ReferenceObject(object_kind="Method", label="IR"),
                ReferenceObject(object_kind="AgenticEntity", label="Bruker ALPHA"),
            ],
            expected_attributes=[
                ReferenceAttribute(title="method", value="IR"),
            ],
            expected_vocab_mappings=[
                ReferenceVocabMapping(
                    source_value="IR",
                    mapping_type="qualitative",
                    acceptable_uris=["http://purl.obolibrary.org/obo/CHMO_0000634"],
                )
            ],
            required_profile_fields=["title"],
        )

        report = evaluate_extraction_result(
            reference=reference,
            result=result,
            state=state,
            schema_valid=True,
        )

        self.assertTrue(report.schema_valid)
        self.assertTrue(report.file_ranking_top1_hit)
        self.assertEqual(report.object_metrics["Method"].recall, 1.0)
        self.assertEqual(report.attribute_metrics.recall, 1.0)
        self.assertEqual(report.vocab_mapping_metrics.recall, 1.0)
        self.assertEqual(report.required_profile_field_coverage, 1.0)

    def test_score_reference_directory_writes_partial_report_for_missing_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            reference_dir = root / "references"
            dataset_dir = root / "datasets"
            output_dir = root / "output"
            results_dir = root / "results"
            reference_dir.mkdir()
            dataset_dir.mkdir()
            dataset_path = dataset_dir / "IR-IR.zip"
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as archive:
                archive.writestr("dataset_description.txt", "IR dataset")
            dataset_path.write_bytes(zip_buffer.getvalue())
            (reference_dir / "IR-IR.json").write_text(
                '{"dataset_filename": "IR-IR.zip", "expected_objects": []}',
                encoding="utf-8",
            )

            reports = score_reference_directory(
                reference_dir=reference_dir,
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                results_dir=results_dir,
            )

            self.assertEqual(reports, [])
            partial_reports = list(results_dir.glob("*/partial_report.json"))
            self.assertEqual(len(partial_reports), 1)
            self.assertIn(
                '"outcome": "missing_output"',
                partial_reports[0].read_text(encoding="utf-8"),
            )
            self.assertIn(
                "missing_output",
                (results_dir / "partial_evaluation_summary.md").read_text(encoding="utf-8"),
            )
            self.assertIn(
                '"dataset_count": 1',
                (results_dir / "manual_reference_baseline.json").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "expected objects: 0",
                (results_dir / "manual_reference_baseline.md").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
