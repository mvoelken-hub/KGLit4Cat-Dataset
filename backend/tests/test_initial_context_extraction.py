import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core.task_registry import TaskRegistry, TaskStatus, TaskType
from app.domain.datasources import ContentChunk, DataPackage, FileEntry
from app.domain.extraction import (
    ChunkingRequiredError,
    DefinedTerm,
    EvidenceContext,
    EvidenceNote,
    ExtractionChunkResult,
    ExtractionContext,
    ExtractionFileSummary,
    ExtractionOverview,
    ExtractionOverviewFilePreview,
    ExtractionNormalization,
    ExtractionRunResult,
    ExtractionRunProgress,
    ExtractionRunState,
    ExtractionVocabQueryConfig,
    ExtractionVocabQueryRecord,
    FilteredEvidenceLedger,
    FileRankingResult,
    GroundedExtractionObject,
    PromptTokenBudgeter,
    ProfilePatchDocument,
    RankedFile,
    Resource,
    TracedExtractionObject,
    VocabularyFallbackQuery,
    VocabularyCandidateSelection,
    VocabularyTermMapping,
)
from app.domain.semantics import CompactVocabResource, VocabQuery, VocabQueryResult, VocabSchemeInfo, VocabTermScheme
from app.ollama.completion import CompletionResult
from app.ollama.errors import MaxRetriesExceeded, OutputParsingError
from app.ollama.usage import RunUsage
from app.services.extraction_service import (
    ExtractionService,
    EXTRACTION_OVERVIEW_SYSTEM_PROMPT,
    _ObjectGroundingCandidateDiscovery,
    _QualitativeCandidateDiscovery,
)


class FakeLogger:
    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def exception(self, *_args, **_kwargs):
        pass


class FakeDataSourceService:
    def __init__(self, chunks_by_file: list[list[ContentChunk]]):
        self.data_package = DataPackage(
            file_name="package",
            files=[
                FileEntry(
                    file_path="README.md",
                    file_name="README.md",
                    file_extension=".md",
                    raw_content=b"metadata",
                )
            ],
        )
        self.chunks_by_file = chunks_by_file
        self.chunk_calls: list[dict] = []
        self.chunk_status = TaskStatus.COMPLETED

    def get_data_package(self, _data_package_id: str) -> DataPackage:
        return self.data_package

    def get_completed_content_chunks_by_file(self, _data_package_id: str):
        return self.chunks_by_file

    def get_chunk_task_status(self, _data_package_id: str) -> TaskStatus:
        if self.chunk_status != TaskStatus.UNKNOWN:
            return self.chunk_status
        return TaskStatus.COMPLETED if self.chunks_by_file else TaskStatus.UNKNOWN

    async def chunk_file_entries_in_data_package(self, **kwargs):
        self.chunk_calls.append(kwargs)
        return self.chunks_by_file, self.chunk_status

    @staticmethod
    def chunk_task_name(data_package_id: str) -> str:
        return f"chunking:file_entries:{data_package_id}"


class FakeProfileService:
    schema = {
        "type": "object",
        "required": ["id"],
        "properties": {"id": {"type": "string"}},
        "additionalProperties": True,
    }

    def get_profile(self, identifier: str):
        return SimpleNamespace(identifier=identifier, target_class="Dataset", enrichable_fields=["type"])

    def load_json_schema(self, _identifier: str):
        return self.schema

    def validate_document(self, *, identifier: str, document: dict):
        errors = [] if isinstance(document.get("id"), str) else [SimpleNamespace(path="$.id", message="required")]
        return SimpleNamespace(valid=not errors, errors=errors)


class TitleProfileService(FakeProfileService):
    schema = {
        "type": "object",
        "required": ["title"],
        "properties": {
            "title": {"type": "string"},
            "identifier": {"type": "string"},
        },
        "additionalProperties": True,
    }

    def validate_document(self, *, identifier: str, document: dict):
        errors = [] if isinstance(document.get("title"), str) else [SimpleNamespace(path="$.title", message="required")]
        return SimpleNamespace(valid=not errors, errors=errors)


class FakeOutputRepository:
    def __init__(self):
        self.context: ExtractionContext | None = None
        self.contexts: list[ExtractionContext] = []
        self.evidence_context: EvidenceContext | None = None
        self.filtered_evidence_notes = FilteredEvidenceLedger()
        self.result: ExtractionRunResult | None = None
        self.run_state: ExtractionRunState | None = None
        self.initial_file_summaries: list[ExtractionFileSummary] = []
        self.initial_file_summary_status = None
        self.initial_file_summary_diagnostics = None
        self.initial_extraction_overview = None
        self.initial_extraction_overview_status = None
        self.initial_extraction_overview_diagnostic = None
        self.generated_final_draft: dict | None = None
        self.curated_document: dict | None = None
        self.projection_ledger: list = []
        self.field_completion_ledger: list = []
        self.curation_ledger: list = []
        self.validation: dict | None = None
        self.warnings: list[str] = []
        self.token_usage: dict[str, dict[str, int]] = {}

    def save_extraction_context(self, *, workflow_id: str, extraction_context: ExtractionContext):
        self.context = extraction_context
        self.contexts.append(extraction_context)

    def load_extraction_context(self, workflow_id: str) -> ExtractionContext:
        if self.context is None:
            raise FileNotFoundError
        return self.context

    def save_evidence_context(self, *, workflow_id: str, evidence_context: EvidenceContext):
        self.evidence_context = evidence_context

    def load_evidence_context(self, workflow_id: str) -> EvidenceContext:
        if self.evidence_context is None:
            raise FileNotFoundError
        return self.evidence_context

    def save_filtered_evidence_notes(self, *, workflow_id: str, ledger: FilteredEvidenceLedger):
        self.filtered_evidence_notes = ledger

    def load_filtered_evidence_notes(self, workflow_id: str) -> FilteredEvidenceLedger:
        return self.filtered_evidence_notes

    def save_extraction_result(self, *, workflow_id: str, result: ExtractionRunResult):
        self.result = result
        self.initial_file_summaries = result.initial_file_summaries
        self.initial_file_summary_status = result.initial_file_summary_status
        self.initial_extraction_overview = result.initial_extraction_overview
        self.initial_extraction_overview_status = result.initial_extraction_overview_status
        self.generated_final_draft = result.generated_final_draft
        self.curated_document = result.curated_document
        self.projection_ledger = result.projection_ledger
        self.field_completion_ledger = result.field_completion_ledger
        self.curation_ledger = result.curation_ledger
        self.validation = {
            "generated": result.validation,
            "curated": result.curated_validation,
        }

    def load_extraction_result(self, workflow_id: str, chat_model: str | None = None) -> ExtractionRunResult:
        if self.result is None:
            raise FileNotFoundError
        return self.result

    def save_initial_file_summaries(
        self,
        *,
        workflow_id: str,
        summaries,
        status,
        chat_model: str | None = None,
    ):
        self.initial_file_summaries = summaries
        self.initial_file_summary_status = status

    def load_initial_file_summaries(self, workflow_id: str, chat_model: str | None = None):
        if self.initial_file_summary_status is None:
            raise FileNotFoundError
        return self.initial_file_summaries, self.initial_file_summary_status

    def save_initial_file_summary_diagnostics(
        self,
        *,
        workflow_id: str,
        diagnostics,
        chat_model: str | None = None,
    ):
        self.initial_file_summary_diagnostics = diagnostics

    def save_initial_extraction_overview(
        self,
        *,
        workflow_id: str,
        overview,
        status,
        chat_model: str | None = None,
    ):
        self.initial_extraction_overview = overview
        self.initial_extraction_overview_status = status

    def load_initial_extraction_overview(self, workflow_id: str, chat_model: str | None = None):
        if self.initial_extraction_overview_status is None:
            raise FileNotFoundError
        return self.initial_extraction_overview, self.initial_extraction_overview_status

    def save_initial_extraction_overview_diagnostic(
        self,
        *,
        workflow_id: str,
        diagnostic,
        chat_model: str | None = None,
    ):
        self.initial_extraction_overview_diagnostic = diagnostic

    def save_extraction_run_state(self, *, workflow_id: str, state: ExtractionRunState):
        self.run_state = state

    def load_extraction_run_state(self, workflow_id: str, chat_model: str | None = None) -> ExtractionRunState:
        if self.run_state is None:
            raise FileNotFoundError
        return self.run_state

    def save_generated_final_draft(self, *, workflow_id: str, document: dict, chat_model: str | None = None):
        self.generated_final_draft = document

    def load_generated_final_draft(self, workflow_id: str, chat_model: str | None = None) -> dict:
        if self.generated_final_draft is None:
            raise FileNotFoundError
        return self.generated_final_draft

    def save_curated_document(self, *, workflow_id: str, document: dict, chat_model: str | None = None):
        self.curated_document = document

    def load_curated_document(self, workflow_id: str, chat_model: str | None = None) -> dict:
        if self.curated_document is None:
            raise FileNotFoundError
        return self.curated_document

    def save_projection_ledger(self, *, workflow_id: str, ledger: list, chat_model: str | None = None):
        self.projection_ledger = ledger

    def load_projection_ledger(self, workflow_id: str, chat_model: str | None = None) -> list:
        return self.projection_ledger

    def save_field_completion_ledger(self, *, workflow_id: str, ledger: list, chat_model: str | None = None):
        self.field_completion_ledger = ledger

    def load_field_completion_ledger(self, workflow_id: str, chat_model: str | None = None) -> list:
        return self.field_completion_ledger

    def save_curation_ledger(self, *, workflow_id: str, ledger: list, chat_model: str | None = None):
        self.curation_ledger = ledger

    def load_curation_ledger(self, workflow_id: str, chat_model: str | None = None) -> list:
        return self.curation_ledger

    def save_validation(
        self,
        *,
        workflow_id: str,
        validation,
        curated_validation=None,
        chat_model: str | None = None,
    ):
        self.validation = {"generated": validation, "curated": curated_validation}

    def load_validation(self, workflow_id: str, chat_model: str | None = None) -> dict:
        if self.validation is None:
            raise FileNotFoundError
        return self.validation

    def save_extraction_warnings(self, *, workflow_id: str, warnings: list[str]):
        self.warnings = warnings

    def load_extraction_warnings(self, workflow_id: str) -> list[str]:
        return self.warnings

    def save_token_usage(self, *, workflow_id: str, token_usage: dict[str, dict[str, int]]):
        self.token_usage = token_usage

    def load_token_usage(self, workflow_id: str) -> dict[str, dict[str, int]]:
        return self.token_usage

    def clear_extraction_run(self, workflow_id: str):
        self.context = None
        self.contexts = []
        self.evidence_context = None
        self.filtered_evidence_notes = FilteredEvidenceLedger()
        self.result = None
        self.run_state = None
        self.initial_file_summaries = []
        self.initial_file_summary_status = None
        self.initial_extraction_overview = None
        self.initial_extraction_overview_status = None
        self.generated_final_draft = None
        self.curated_document = None
        self.projection_ledger = []
        self.field_completion_ledger = []
        self.curation_ledger = []
        self.validation = None
        self.warnings = []
        self.token_usage = {}

    def clear_extraction_downstream(self, workflow_id: str):
        self.context = None
        self.contexts = []
        self.evidence_context = None
        self.filtered_evidence_notes = FilteredEvidenceLedger()
        self.result = None
        if self.run_state is not None:
            self.run_state = self.run_state.model_copy(
                update={
                    "chunk_results": [],
                    "vocab_queries": [],
                    "generated_final_draft": None,
                    "curated_document": None,
                    "draft_quality_state": None,
                    "projection_ledger": [],
                    "field_completion_ledger": [],
                    "curation_ledger": [],
                }
            )
        self.generated_final_draft = None
        self.curated_document = None
        self.projection_ledger = []
        self.field_completion_ledger = []
        self.curation_ledger = []
        self.validation = None


def make_chunk(start_idx: int = 0, content: str = "sample measured at 20 C") -> ContentChunk:
    return ContentChunk(
        content=content,
        data_package_id="package-id",
        file_path="README.md",
        start_idx=start_idx,
        end_idx=start_idx,
    )


def overview_for_file(
    file_path: str = "README.md",
    *,
    source_fingerprint: str = "",
    summary: str = "File is present.",
    uncertainty: str = "Use graph orientation only.",
) -> ExtractionOverview:
    return ExtractionOverview(
        source_fingerprint=source_fingerprint,
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
        uncertainties=[uncertainty] if uncertainty else [],
    )


def resource_context(identifier: str, description: str) -> ExtractionContext:
    return ExtractionContext.model_validate(
        {
            "extraction_objects": [
                {
                    "object_kind": "Resource",
                    "extracted_object": {
                        "identifier": identifier,
                        "type": "dataset",
                        "description": description,
                    },
                    "source_text": description,
                }
            ]
        }
    )


def qualitative_context(identifier: str, title: str, value: str) -> ExtractionContext:
    return ExtractionContext.model_validate(
        {
            "extraction_objects": [
                {
                    "object_kind": "Resource",
                    "extracted_object": {
                        "identifier": identifier,
                        "type": "dataset",
                        "description": f"{title} {value}",
                        "has_qualitative_attributes": [
                            {"title": title, "value": value}
                        ],
                    },
                    "source_text": f"{title} {value}",
                }
            ]
        }
    )


def quantitative_context(identifier: str, quantity_identifier: str = "temperature") -> ExtractionContext:
    return ExtractionContext.model_validate(
        {
            "extraction_objects": [
                {
                    "object_kind": "Resource",
                    "extracted_object": {
                        "identifier": identifier,
                        "type": "dataset",
                        "description": "Temperature measurement.",
                        "has_quantitative_attributes": [
                            {
                                "identifier": quantity_identifier,
                                "value": "300",
                                "unit": "K",
                                "quantity_kind": "temperature",
                            }
                        ],
                    },
                    "source_text": "Temperature 300 K",
                }
            ]
        }
    )


def empty_profile_patch() -> CompletionResult[ProfilePatchDocument]:
    return CompletionResult(
        output=ProfilePatchDocument(),
        usage=RunUsage(requests=1),
    )


def type_profile_patch(value: str = "dataset") -> CompletionResult[ProfilePatchDocument]:
    return CompletionResult(
        output=ProfilePatchDocument(
            operations=[{"op": "add", "path": "/type", "value": value}]
        ),
        usage=RunUsage(requests=1),
    )


def quantity_profile_patch() -> CompletionResult[ProfilePatchDocument]:
    return CompletionResult(
        output=ProfilePatchDocument(
            operations=[
                {"op": "add", "path": "/has_quantity_type", "value": "temperature"},
                {"op": "add", "path": "/unit", "value": "K"},
            ]
        ),
        usage=RunUsage(requests=1),
    )


class FakeSemanticService:
    def __init__(self):
        self.discovery_started = asyncio.Event()
        self.query_calls: list[str] = []

    async def get_vocabulary(self, identifier: str):
        return VocabSchemeInfo(
            identifier=identifier,
            source=identifier,
            rdf_format="text/turtle",
            num_triples=1,
            vocab_term_schemes=[
                VocabTermScheme(
                    rdf_type="skos__Concept",
                    properties=["skos__prefLabel"],
                    count=1,
                )
            ],
        )

    async def query_vocabulary(self, identifier: str, query):
        self.query_calls.append(f"{identifier}:{query.rdf_type}")
        self.discovery_started.set()
        await asyncio.sleep(0.05)
        return VocabQueryResult(
            identifier=identifier,
            rdf_type=query.rdf_type,
            resources={},
        )


def make_service(
    chunks_by_file: list[list[ContentChunk]],
    *,
    with_initial_context: bool = True,
):
    task_registry = TaskRegistry(SimpleNamespace(), FakeLogger())  # type: ignore[arg-type]
    output_repository = FakeOutputRepository()
    if with_initial_context:
        output_repository.run_state = ExtractionRunState(
            chat_model="chat",
            ranked_files=[RankedFile(rank=1, file_path="README.md")],
            initial_file_summaries=[
                ExtractionFileSummary(
                    file_path="README.md",
                    data_format="markdown",
                )
            ],
            initial_file_summary_status="completed",
            initial_extraction_overview=overview_for_file(
                source_fingerprint="overview",
                summary="README.md is present.",
            ),
            initial_extraction_overview_status="structured",
        )
    service = ExtractionService(
        FakeProfileService(),  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        FakeDataSourceService(chunks_by_file),  # type: ignore[arg-type]
        SimpleNamespace(chat_model="chat", max_context_length=4096),  # type: ignore[arg-type]
        output_repository,
        task_registry,
        semantic_service=None,
    )
    return service, task_registry, output_repository


class ExtractionServiceWorkflowTests(unittest.IsolatedAsyncioTestCase):
    class WhitespaceEncoding:
        def __init__(self, text: str):
            import re

            matches = list(re.finditer(r"\S+", text))
            self.ids = list(range(len(matches)))
            self.offsets = [(match.start(), match.end()) for match in matches]

    class WhitespaceTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            return ExtractionServiceWorkflowTests.WhitespaceEncoding(text)

    async def test_initial_context_run_does_not_require_chunks_or_profile(self):
        service, task_registry, output_repository = make_service(
            [],
            with_initial_context=False,
        )

        status = await service.run_initial_context(data_package_id="package-id")

        self.assertEqual(status, TaskStatus.RUNNING)
        await task_registry.wait_for_task("initial-context:package-id", timeout=2)
        progress_status, progress = await service.get_initial_context_progress(
            data_package_id="package-id"
        )

        self.assertEqual(progress_status, TaskStatus.COMPLETED)
        self.assertIsNotNone(progress)
        self.assertEqual(progress.stage, "initial_context_completed")
        self.assertEqual(output_repository.initial_file_summary_status, "failed")
        self.assertEqual(output_repository.initial_extraction_overview_status, "failed")
        self.assertEqual(len(output_repository.initial_file_summaries), 1)

    async def test_run_extraction_requires_initial_context_artifacts(self):
        service, task_registry, _ = make_service(
            [[make_chunk()]],
            with_initial_context=False,
        )

        _, status = await service.run_extraction(
            data_package_id="package-id",
            profile_identifier="profile",
        )

        self.assertEqual(status, TaskStatus.RUNNING)
        with self.assertRaises(ValueError):
            await task_registry.wait_for_task("extraction:run:package-id", timeout=2)

    async def test_initial_overview_previews_use_ranked_top_files_and_raw_first_lines(self):
        service, _, _ = make_service([[make_chunk()]])
        data_package = DataPackage(
            file_name="package",
            files=[
                FileEntry(
                    file_path="a.txt",
                    file_name="a.txt",
                    file_extension=".txt",
                    raw_content="\n".join(f"a-{i}" for i in range(100)).encode(),
                ),
                FileEntry(
                    file_path="b.txt",
                    file_name="b.txt",
                    file_extension=".txt",
                    raw_content=b"b-0\nb-1",
                ),
                FileEntry(
                    file_path="c.txt",
                    file_name="c.txt",
                    file_extension=".txt",
                    raw_content=b"c-0",
                ),
                FileEntry(
                    file_path="plot.png",
                    file_name="plot.png",
                    file_extension=".png",
                    raw_content=b"image-bytes",
                ),
            ],
        )
        ranking = FileRankingResult(files=[
            RankedFile(rank=1, file_path="b.txt"),
            RankedFile(rank=2, file_path="plot.png"),
            RankedFile(rank=3, file_path="a.txt"),
            RankedFile(rank=4, file_path="c.txt"),
        ])

        previews = service._initial_overview_file_previews(
            data_package=data_package,
            ranking=ranking,
        )

        self.assertEqual([preview.file_path for preview in previews], ["b.txt", "plot.png", "a.txt", "c.txt"])
        self.assertEqual(previews[0].first_lines, ["b-0", "b-1"])
        self.assertEqual(previews[1].first_lines, [])
        self.assertEqual(len(previews[2].first_lines), 80)
        self.assertEqual(previews[2].first_lines[0], "a-0")
        self.assertEqual(previews[2].first_lines[-1], "a-79")

    async def test_initial_file_summaries_use_budgeted_windows_and_sanitize_purpose(self):
        service, _, output_repository = make_service([[make_chunk()]])
        service.ollama_client.ollama_client = SimpleNamespace()  # type: ignore[attr-defined]
        content = (
            "BEGIN JCAMP-DX ##TITLE=HMS-Q11-p\n"
            + ("middle spectral data\n" * 800)
            + "END ##$PULPROG=zg30 ##$PLW1=12\n"
        )
        data_package = DataPackage(
            file_name="nmr-package",
            files=[
                FileEntry(
                    file_path="HMS-Q11-p_10.dx",
                    file_name="HMS-Q11-p_10.dx",
                    file_extension=".dx",
                    raw_content=content.encode(),
                )
            ],
        )
        state = ExtractionRunState(profile_identifier="profile")

        async def fake_generate(*_args, **kwargs):
            self.assertIs(kwargs["output_type"], ExtractionFileSummary)
            self.assertIn('"label":"beginning"', kwargs["prompt"])
            self.assertIn('"label":"middle"', kwargs["prompt"])
            self.assertIn('"label":"end"', kwargs["prompt"])
            self.assertIn("common metadata categories", kwargs["prompt"])
            self.assertIn("These categories are examples only", kwargs["prompt"])
            self.assertIn("quantitative_signals", kwargs["prompt"])
            self.assertNotIn("parameter_terms", kwargs["prompt"])
            return CompletionResult(
                output=ExtractionFileSummary(
                    file_path="made-up.py",
                    data_format="JCAMP-DX spectroscopy export",
                    explicit_purpose="Stores final NMR evidence.",
                    instrument_or_software_terms_and_settings=["PULPROG=zg30"],
                    quantitative_signals=["explicit numeric settings"],
                ),
                usage=RunUsage(requests=1),
            )

        warnings: list[str] = []
        with patch("app.services.extraction_service.generate_structured", side_effect=fake_generate):
            await service._generate_initial_file_summaries(
                data_package_id="package-id",
                data_package=data_package,
                state=state,
                warnings=warnings,
            )

        self.assertEqual(state.initial_file_summary_status, "completed")
        self.assertEqual(len(state.initial_file_summaries), 1)
        summary = state.initial_file_summaries[0]
        self.assertEqual(summary.file_path, "HMS-Q11-p_10.dx")
        self.assertEqual(summary.explicit_purpose, "")
        self.assertIn("PULPROG=zg30", summary.instrument_or_software_terms_and_settings)
        self.assertIn("explicit numeric settings", summary.quantitative_signals)
        self.assertIsNotNone(output_repository.initial_file_summary_diagnostics)
        self.assertEqual(output_repository.initial_file_summaries, state.initial_file_summaries)
        self.assertIsNotNone(state.initial_file_summary_progress)
        self.assertEqual(state.initial_file_summary_progress.total_files, 1)
        self.assertEqual(state.initial_file_summary_progress.processed_files, 1)
        self.assertEqual(state.initial_file_summary_progress.summarized_files, 1)
        self.assertIsNone(state.initial_file_summary_progress.current_file_path)
        self.assertTrue(any("mismatched file_path" in warning for warning in warnings))
        self.assertTrue(any("without evidence" in warning for warning in warnings))

    async def test_initial_file_summaries_skip_images(self):
        service, _, output_repository = make_service([[make_chunk()]])
        service.ollama_client.ollama_client = SimpleNamespace()  # type: ignore[attr-defined]
        data_package = DataPackage(
            file_name="nmr-package",
            files=[
                FileEntry(
                    file_path="spectrum.png",
                    file_name="spectrum.png",
                    file_extension=".png",
                    raw_content=b"image-bytes",
                ),
                FileEntry(
                    file_path="notes.txt",
                    file_name="notes.txt",
                    file_extension=".txt",
                    raw_content=b"instrument: Bruker Alpha-P ATR",
                ),
            ],
        )
        state = ExtractionRunState(profile_identifier="profile")
        calls: list[dict] = []

        async def fake_generate(*_args, **kwargs):
            calls.append(kwargs)
            return CompletionResult(
                output=ExtractionFileSummary(
                    file_path="notes.txt",
                    data_format="text",
                    metadata_signals=["instrument: Bruker Alpha-P ATR"],
                ),
                usage=RunUsage(requests=1),
            )

        warnings: list[str] = []
        with patch("app.services.extraction_service.generate_structured", side_effect=fake_generate):
            await service._generate_initial_file_summaries(
                data_package_id="package-id",
                data_package=data_package,
                state=state,
                warnings=warnings,
            )

        self.assertEqual(state.initial_file_summary_status, "completed")
        self.assertEqual([summary.file_path for summary in state.initial_file_summaries], ["notes.txt"])
        self.assertEqual(len(calls), 1)
        self.assertIn("notes.txt", calls[0]["prompt"])
        self.assertNotIn("spectrum.png", calls[0]["prompt"])
        self.assertEqual(output_repository.initial_file_summaries, state.initial_file_summaries)
        self.assertTrue(any("spectrum.png" in warning and "skipped" in warning for warning in warnings))

    async def test_initial_overview_generation_stamps_file_provenance(self):
        service, _, output_repository = make_service([[make_chunk()]])
        service.ollama_client.ollama_client = SimpleNamespace()  # type: ignore[attr-defined]
        data_package = DataPackage(
            file_name="nmr-package",
            files=[
                FileEntry(
                    file_path="dataset_description.txt",
                    file_name="dataset_description.txt",
                    file_extension=".txt",
                    raw_content=b"1H NMR spectrum for sample HMS-Q11-p.",
                ),
                FileEntry(
                    file_path="HMS-Q11-p_10.dx",
                    file_name="HMS-Q11-p_10.dx",
                    file_extension=".dx",
                    raw_content=b"##TITLE=HMS-Q11-p\n##.OBSERVE NUCLEUS=1H\n##$PULPROG=zg30",
                ),
            ],
        )
        ranking = FileRankingResult(files=[
            RankedFile(rank=1, file_path="dataset_description.txt"),
            RankedFile(rank=2, file_path="HMS-Q11-p_10.dx"),
        ])
        state = ExtractionRunState(profile_identifier="profile")

        async def fake_generate(*_args, **kwargs):
            self.assertIs(kwargs["output_type"], ExtractionOverview)
            self.assertIn("scientific data archivist", kwargs["system"])
            self.assertIn("file-centered package graph", kwargs["system"])
            self.assertIn("nodes", kwargs["system"])
            self.assertIn("edges", kwargs["system"])
            self.assertIn("uncertainties", kwargs["system"])
            self.assertIn("controlled relation", kwargs["system"])
            self.assertIn("Do not create contains edges", kwargs["system"])
            self.assertIn("Ranked order is prioritization metadata only", kwargs["system"])
            self.assertIn("First group files by shared package role", kwargs["system"])
            self.assertIn("connect files to group nodes first", kwargs["system"])
            self.assertIn("package graph triage, not final scientific interpretation", kwargs["prompt"])
            self.assertIn("Backend-seeded package/directory/file/group graph JSON", kwargs["prompt"])
            self.assertIn("Ranked order means extraction priority only", kwargs["prompt"])
            self.assertIn("Group files first", kwargs["prompt"])
            self.assertIn("dataset_description.txt", kwargs["prompt"])
            self.assertIn("##$PULPROG=zg30", kwargs["prompt"])
            return CompletionResult(
                output=ExtractionOverview(
                    nodes=[
                        {
                            "node_id": "group:raw-data",
                            "label": "raw data",
                            "kind": "group",
                        },
                    ],
                    edges=[
                        {
                            "source": "file:HMS-Q11-p_10.dx",
                            "target": "group:raw-data",
                            "relation": "describes",
                            "evidence": ["##.OBSERVE NUCLEUS=1H"],
                            "note": "Raw data grouping is suggested by file summary.",
                        }
                    ],
                    uncertainties=[
                        "Do not resolve instrument or method identity without chunk evidence."
                    ],
                ),
                usage=RunUsage(requests=1),
            )

        with patch("app.services.extraction_service.generate_structured", side_effect=fake_generate):
            await service._generate_initial_extraction_overview(
                data_package_id="package-id",
                data_package=data_package,
                ranking=ranking,
                state=state,
                warnings=[],
            )

        self.assertEqual(state.initial_extraction_overview_status, "structured")
        overview = state.initial_extraction_overview
        self.assertIsNotNone(overview)
        self.assertEqual(
            overview.source_file_paths,
            ["dataset_description.txt", "HMS-Q11-p_10.dx"],
        )
        self.assertTrue(overview.source_fingerprint)
        self.assertEqual(
            [file.file_path for file in overview.inspected_files],
            ["dataset_description.txt", "HMS-Q11-p_10.dx"],
        )
        self.assertEqual(
            [node.file_path for node in overview.nodes if node.kind == "file"],
            ["dataset_description.txt", "HMS-Q11-p_10.dx"],
        )
        self.assertIn("package:root", {node.node_id for node in overview.nodes})
        self.assertEqual(overview.edges[0].relation, "contains")
        self.assertTrue(
            any(
                edge.source == "file:HMS-Q11-p_10.dx"
                and edge.target == "group:raw-data"
                and edge.relation == "describes"
                for edge in overview.edges
            )
        )
        self.assertEqual(
            overview.uncertainties,
            ["Do not resolve instrument or method identity without chunk evidence."],
        )

    def test_initial_overview_seed_graph_creates_path_containment(self):
        seed = ExtractionService._seed_initial_overview_graph(
            data_package_name="package",
            ranked_files=[
                RankedFile(rank=1, file_path="metadata.txt"),
                RankedFile(rank=2, file_path="raw/run1/data.txt"),
            ],
            file_summaries=[
                ExtractionFileSummary(
                    file_path="raw/run1/data.txt",
                    data_format="text",
                    metadata_signals=["raw measurements"],
                )
            ],
            file_previews=[
                ExtractionOverviewFilePreview(rank=1, file_path="metadata.txt"),
                ExtractionOverviewFilePreview(rank=2, file_path="raw/run1/data.txt"),
            ],
        )

        node_ids = {node.node_id for node in seed.nodes}
        self.assertIn("package:root", node_ids)
        self.assertIn("dir:raw", node_ids)
        self.assertIn("dir:raw/run1", node_ids)
        self.assertIn("file:metadata.txt", node_ids)
        self.assertIn("file:raw/run1/data.txt", node_ids)
        edge_keys = {(edge.source, edge.relation, edge.target) for edge in seed.edges}
        self.assertIn(("package:root", "contains", "file:metadata.txt"), edge_keys)
        self.assertIn(("package:root", "contains", "dir:raw"), edge_keys)
        self.assertIn(("dir:raw", "contains", "dir:raw/run1"), edge_keys)
        self.assertIn(("dir:raw/run1", "contains", "file:raw/run1/data.txt"), edge_keys)

    def test_initial_overview_seed_graph_creates_summary_supported_groups(self):
        seed = ExtractionService._seed_initial_overview_graph(
            data_package_name="package",
            ranked_files=[
                RankedFile(rank=1, file_path="dataset_description.txt"),
                RankedFile(rank=2, file_path="audit/log.txt"),
                RankedFile(rank=3, file_path="method/program.txt"),
                RankedFile(rank=4, file_path="settings/acquisition.txt"),
                RankedFile(rank=5, file_path="settings/processing.txt"),
                RankedFile(rank=6, file_path="settings/instrument.txt"),
                RankedFile(rank=7, file_path="data/raw.txt"),
                RankedFile(rank=8, file_path="data/processed.txt"),
            ],
            file_summaries=[
                ExtractionFileSummary(
                    file_path="dataset_description.txt",
                    explicit_purpose="dataset description",
                    metadata_signals=["human-readable"],
                ),
                ExtractionFileSummary(
                    file_path="audit/log.txt",
                    metadata_signals=["audit trail", "hash values"],
                ),
                ExtractionFileSummary(
                    file_path="method/program.txt",
                    instrument_or_software_terms_and_settings=["pulse sequence"],
                ),
                ExtractionFileSummary(
                    file_path="settings/acquisition.txt",
                    explicit_purpose="acquisition parameter configuration",
                    metadata_signals=["settings"],
                    quantitative_signals=["explicit numeric settings"],
                ),
                ExtractionFileSummary(
                    file_path="settings/processing.txt",
                    explicit_purpose="processing parameter file",
                    metadata_signals=["parameter file"],
                ),
                ExtractionFileSummary(
                    file_path="settings/instrument.txt",
                    instrument_or_software_terms_and_settings=[
                        "instrument settings",
                        "calibration",
                    ],
                ),
                ExtractionFileSummary(
                    file_path="data/raw.txt",
                    metadata_signals=["raw measurements"],
                ),
                ExtractionFileSummary(
                    file_path="data/processed.txt",
                    metadata_signals=["processed data"],
                ),
            ],
            file_previews=[
                ExtractionOverviewFilePreview(rank=1, file_path="dataset_description.txt"),
                ExtractionOverviewFilePreview(rank=2, file_path="audit/log.txt"),
                ExtractionOverviewFilePreview(rank=3, file_path="method/program.txt"),
                ExtractionOverviewFilePreview(rank=4, file_path="settings/acquisition.txt"),
                ExtractionOverviewFilePreview(rank=5, file_path="settings/processing.txt"),
                ExtractionOverviewFilePreview(rank=6, file_path="settings/instrument.txt"),
                ExtractionOverviewFilePreview(rank=7, file_path="data/raw.txt"),
                ExtractionOverviewFilePreview(rank=8, file_path="data/processed.txt"),
            ],
        )

        node_ids = {node.node_id for node in seed.nodes}
        self.assertIn("group:dataset_documentation", node_ids)
        self.assertIn("group:audit_provenance", node_ids)
        self.assertIn("group:method_program", node_ids)
        self.assertIn("group:acquisition_settings", node_ids)
        self.assertIn("group:processing_settings", node_ids)
        self.assertIn("group:instrument_settings", node_ids)
        self.assertIn("group:raw_data", node_ids)
        self.assertIn("group:processed_data", node_ids)
        self.assertNotIn("group:parameter_settings", node_ids)

        edge_keys = {(edge.source, edge.relation, edge.target) for edge in seed.edges}
        self.assertIn(
            (
                "file:dataset_description.txt",
                "documents",
                "group:dataset_documentation",
            ),
            edge_keys,
        )
        self.assertIn(
            ("file:settings/acquisition.txt", "parameterizes", "group:acquisition_settings"),
            edge_keys,
        )
        self.assertIn(
            ("file:method/program.txt", "configures", "group:method_program"),
            edge_keys,
        )
        self.assertIn(
            ("group:method_program", "configures", "group:acquisition_settings"),
            edge_keys,
        )
        self.assertIn(
            ("group:processed_data", "derives_from", "group:raw_data"),
            edge_keys,
        )

    def test_initial_overview_seed_graph_ignores_negated_group_signals(self):
        seed = ExtractionService._seed_initial_overview_graph(
            data_package_name="package",
            ranked_files=[
                RankedFile(rank=1, file_path="not-raw.txt"),
                RankedFile(rank=2, file_path="minimal.txt"),
            ],
            file_summaries=[
                ExtractionFileSummary(
                    file_path="not-raw.txt",
                    metadata_signals=["support file format, not raw data"],
                ),
                ExtractionFileSummary(
                    file_path="minimal.txt",
                    metadata_signals=["minimal content, no detailed parameters"],
                ),
            ],
            file_previews=[
                ExtractionOverviewFilePreview(rank=1, file_path="not-raw.txt"),
                ExtractionOverviewFilePreview(rank=2, file_path="minimal.txt"),
            ],
        )

        node_ids = {node.node_id for node in seed.nodes}
        self.assertNotIn("group:raw_data", node_ids)
        self.assertNotIn("group:parameter_settings", node_ids)

    def test_initial_overview_sanitizer_removes_ranked_star_containment(self):
        seed = ExtractionService._seed_initial_overview_graph(
            data_package_name="package",
            ranked_files=[
                RankedFile(rank=1, file_path="10.infer.json"),
                RankedFile(rank=2, file_path="dataset_description.txt"),
            ],
            file_summaries=[],
            file_previews=[
                ExtractionOverviewFilePreview(rank=1, file_path="10.infer.json"),
                ExtractionOverviewFilePreview(rank=2, file_path="dataset_description.txt"),
            ],
        )
        warnings: list[str] = []

        sanitized = ExtractionService._sanitize_initial_overview_graph(
            ExtractionOverview(
                nodes=[
                    {
                        "node_id": "file:10.infer.json",
                        "label": "10.infer.json",
                        "kind": "file",
                        "file_path": "10.infer.json",
                    },
                    {
                        "node_id": "file:dataset_description.txt",
                        "label": "dataset_description.txt",
                        "kind": "file",
                        "file_path": "dataset_description.txt",
                    },
                ],
                edges=[
                    {
                        "edge_id": "bad-star-edge",
                        "source": "file:10.infer.json",
                        "target": "file:dataset_description.txt",
                        "relation": "contains",
                        "evidence": ["ranked"],
                    }
                ],
            ),
            allowed_file_paths={"10.infer.json", "dataset_description.txt"},
            seed_overview=seed,
            warnings=warnings,
        )

        edge_keys = {(edge.source, edge.relation, edge.target) for edge in sanitized.edges}
        self.assertNotIn(
            ("file:10.infer.json", "contains", "file:dataset_description.txt"),
            edge_keys,
        )
        self.assertIn(("package:root", "contains", "file:10.infer.json"), edge_keys)
        self.assertIn(("package:root", "contains", "file:dataset_description.txt"), edge_keys)
        self.assertTrue(any("non-path containment" in warning for warning in warnings))

    def test_initial_overview_sanitizer_drops_placeholder_semantic_evidence(self):
        seed = ExtractionService._seed_initial_overview_graph(
            data_package_name="package",
            ranked_files=[
                RankedFile(rank=1, file_path="settings.txt"),
                RankedFile(rank=2, file_path="data.txt"),
            ],
            file_summaries=[],
            file_previews=[
                ExtractionOverviewFilePreview(rank=1, file_path="settings.txt"),
                ExtractionOverviewFilePreview(rank=2, file_path="data.txt"),
            ],
        )
        warnings: list[str] = []

        sanitized = ExtractionService._sanitize_initial_overview_graph(
            ExtractionOverview(
                edges=[
                    {
                        "source": "file:settings.txt",
                        "target": "file:data.txt",
                        "relation": "configures",
                        "evidence": ["ranked"],
                    },
                    {
                        "source": "file:settings.txt",
                        "target": "file:data.txt",
                        "relation": "documents",
                        "evidence": ["parameter labels"],
                    },
                ],
            ),
            allowed_file_paths={"settings.txt", "data.txt"},
            seed_overview=seed,
            warnings=warnings,
        )

        semantic_edges = [
            edge for edge in sanitized.edges
            if edge.relation != "contains"
        ]
        self.assertEqual(len(semantic_edges), 1)
        self.assertEqual(semantic_edges[0].relation, "documents")
        self.assertEqual(semantic_edges[0].evidence, ["parameter labels"])
        self.assertTrue(any("weak placeholder evidence" in warning for warning in warnings))

    def test_initial_overview_sanitizer_removes_false_missing_summary_uncertainty(self):
        seed = ExtractionService._seed_initial_overview_graph(
            data_package_name="package",
            ranked_files=[RankedFile(rank=1, file_path="settings.txt")],
            file_summaries=[
                ExtractionFileSummary(
                    file_path="settings.txt",
                    data_format="text",
                    metadata_signals=["settings"],
                )
            ],
            file_previews=[
                ExtractionOverviewFilePreview(rank=1, file_path="settings.txt"),
            ],
        )
        warnings: list[str] = []

        sanitized = ExtractionService._sanitize_initial_overview_graph(
            ExtractionOverview(
                uncertainties=[
                    "No per-file summaries available to extract additional semantic relations.",
                    "Processing/raw relation remains ambiguous.",
                ]
            ),
            allowed_file_paths={"settings.txt"},
            seed_overview=seed,
            summaries_available=True,
            warnings=warnings,
        )

        self.assertEqual(
            sanitized.uncertainties,
            ["Processing/raw relation remains ambiguous."],
        )
        self.assertTrue(any("contradicted by available file summaries" in warning for warning in warnings))

    async def test_initial_overview_skips_images_from_previews_and_provenance(self):
        service, _, output_repository = make_service([[make_chunk()]])
        service.ollama_client.ollama_client = SimpleNamespace()  # type: ignore[attr-defined]
        data_package = DataPackage(
            file_name="image-package",
            files=[
                FileEntry(
                    file_path="spectrum.png",
                    file_name="spectrum.png",
                    file_extension=".png",
                    raw_content=b"image-bytes",
                ),
                FileEntry(
                    file_path="notes.txt",
                    file_name="notes.txt",
                    file_extension=".txt",
                    raw_content=b"measurement notes",
                ),
            ],
        )
        ranking = FileRankingResult(files=[
            RankedFile(rank=1, file_path="notes.txt"),
        ])
        state = ExtractionRunState(
            profile_identifier="profile",
            initial_file_summaries=[
                ExtractionFileSummary(file_path="notes.txt"),
            ],
            initial_file_summary_status="completed",
        )

        async def fake_generate(*_args, **kwargs):
            self.assertIn("spectrum.png", kwargs["prompt"])
            self.assertIn("notes.txt", kwargs["prompt"])
            return CompletionResult(
                output=overview_for_file("notes.txt", summary="notes.txt contains text notes."),
                usage=RunUsage(requests=1),
            )

        with patch("app.services.extraction_service.generate_structured", side_effect=fake_generate):
            await service._generate_initial_extraction_overview(
                data_package_id="package-id",
                data_package=data_package,
                ranking=ranking,
                state=state,
                warnings=[],
            )

        overview = state.initial_extraction_overview
        self.assertIsNotNone(overview)
        self.assertEqual(overview.source_file_paths, ["notes.txt", "spectrum.png"])
        self.assertEqual(
            [file.file_path for file in overview.inspected_files],
            ["notes.txt", "spectrum.png"],
        )
        self.assertEqual(output_repository.initial_extraction_overview, overview)

    async def test_initial_overview_compacts_summaries_before_structured_call(self):
        service, _, _ = make_service([[make_chunk()]])
        service.ollama_client.ollama_client = SimpleNamespace()  # type: ignore[attr-defined]
        service.settings.ollama_chat_tokenizer = "fake/tokenizer"
        budgeter = PromptTokenBudgeter(tokenizer=self.WhitespaceTokenizer())
        data_package = DataPackage(
            file_name="large-package",
            files=[
                FileEntry(
                    file_path="parameters.txt",
                    file_name="parameters.txt",
                    file_extension=".txt",
                    raw_content=b"metadata",
                )
            ],
        )
        ranking = FileRankingResult(files=[RankedFile(rank=1, file_path="parameters.txt")])
        state = ExtractionRunState(
            profile_identifier="profile",
            initial_file_summaries=[
                ExtractionFileSummary(
                    file_path="parameters.txt",
                    data_format="plain text",
                    metadata_signals=[f"metadata signal {index}" for index in range(40)],
                    quantitative_signals=[
                        f"quantitative signal {index}" for index in range(120)
                    ],
                )
            ],
            initial_file_summary_status="completed",
        )

        async def fake_generate(*_args, **kwargs):
            prompt = kwargs["prompt"]
            self.assertIn("quantitative signal 0", prompt)
            self.assertNotIn("quantitative signal 119", prompt)
            self.assertLessEqual(
                budgeter.count(EXTRACTION_OVERVIEW_SYSTEM_PROMPT + prompt),
                service._initial_overview_input_token_budget(
                    service.ollama_client.max_context_length,
                ),
            )
            return CompletionResult(
                output=overview_for_file("parameters.txt", summary="parameters.txt was summarized."),
                usage=RunUsage(requests=1, input_tokens=100, output_tokens=20),
            )

        with (
            patch("app.services.extraction_service.generate_structured", side_effect=fake_generate),
            patch(
                "app.services.extraction_service.PromptTokenBudgeter.from_tokenizer_source",
                return_value=budgeter,
            ) as tokenizer_loader,
        ):
            await service._generate_initial_extraction_overview(
                data_package_id="package-id",
                data_package=data_package,
                ranking=ranking,
                state=state,
                warnings=[],
            )

        tokenizer_loader.assert_called_once_with("fake/tokenizer")
        self.assertEqual(state.initial_extraction_overview_status, "structured")

    async def test_initial_overview_structured_failure_persists_diagnostic(self):
        service, _, output_repository = make_service([[make_chunk()]])
        budgeter = PromptTokenBudgeter(tokenizer=self.WhitespaceTokenizer())

        class FakeOllamaGenerate:
            async def generate(self, **_kwargs):
                return SimpleNamespace(
                    response="Fallback overview text.",
                    prompt_eval_count=12,
                    eval_count=8,
                    prompt_eval_duration=0,
                    load_duration=0,
                    eval_duration=0,
                    total_duration=0,
                )

        service.ollama_client.ollama_client = FakeOllamaGenerate()  # type: ignore[attr-defined]
        service.settings.ollama_chat_tokenizer = "fake/tokenizer"
        data_package = DataPackage(
            file_name="diagnostic-package",
            files=[
                FileEntry(
                    file_path="metadata.txt",
                    file_name="metadata.txt",
                    file_extension=".txt",
                    raw_content=b"metadata",
                )
            ],
        )
        ranking = FileRankingResult(files=[RankedFile(rank=1, file_path="metadata.txt")])
        state = ExtractionRunState(
            profile_identifier="profile",
            initial_file_summaries=[
                ExtractionFileSummary(file_path="metadata.txt", data_format="plain text")
            ],
            initial_file_summary_status="completed",
        )

        async def fake_generate(*_args, **_kwargs):
            raise MaxRetriesExceeded(
                last_error=OutputParsingError("bad json"),
                failed_response='{"nodes": [{"node_id": "unterminated"',
                usage=RunUsage(requests=2, input_tokens=200, output_tokens=50),
            )

        with (
            patch("app.services.extraction_service.generate_structured", side_effect=fake_generate),
            patch(
                "app.services.extraction_service.PromptTokenBudgeter.from_tokenizer_source",
                return_value=budgeter,
            ),
        ):
            await service._generate_initial_extraction_overview(
                data_package_id="package-id",
                data_package=data_package,
                ranking=ranking,
                state=state,
                warnings=[],
            )

        self.assertEqual(state.initial_extraction_overview_status, "unstructured_fallback")
        diagnostic = output_repository.initial_extraction_overview_diagnostic
        self.assertIsNotNone(diagnostic)
        self.assertEqual(diagnostic.error_type, "MaxRetriesExceeded")
        self.assertEqual(diagnostic.last_error_type, "OutputParsingError")
        self.assertIn("bad json", diagnostic.last_error)
        self.assertIn("unterminated", diagnostic.failed_response_excerpt)
        self.assertEqual(diagnostic.usage["requests"], 2)
        self.assertIn("total_input_tokens", diagnostic.prompt_budget)

    async def test_prepare_run_state_does_not_reuse_old_chunks_without_overview(self):
        service, _, _ = make_service([[make_chunk()]])
        chunk = make_chunk(content="new content")
        ranking = FileRankingResult(files=[RankedFile(rank=1, file_path=chunk.file_path)])
        old_context = resource_context("old", "Old context")
        persisted_without_overview = ExtractionRunState(
            chunk_results=[
                ExtractionChunkResult(
                    chunk_index=0,
                    file_path=chunk.file_path,
                    start_idx=chunk.start_idx,
                    end_idx=chunk.end_idx,
                    status="completed",
                    extraction_context=old_context,
                )
            ]
        )

        state = service._prepare_run_state(
            ranking=ranking,
            ordered_chunks=[chunk],
            persisted_state=persisted_without_overview,
            profile_identifier="profile",
            vocab_query_config=ExtractionVocabQueryConfig(),
        )

        self.assertEqual(state.chunk_results[0].status, "pending")
        self.assertIsNone(state.chunk_results[0].extraction_context)

        persisted_with_overview = persisted_without_overview.model_copy(
            update={
                "initial_file_summaries": [
                    ExtractionFileSummary(
                        file_path=chunk.file_path,
                        data_format="text",
                    )
                ],
                "initial_file_summary_status": "completed",
                "initial_extraction_overview": overview_for_file(
                    chunk.file_path,
                    summary=f"{chunk.file_path} is present.",
                ),
                "initial_extraction_overview_status": "structured",
            }
        )
        resumed = service._prepare_run_state(
            ranking=ranking,
            ordered_chunks=[chunk],
            persisted_state=persisted_with_overview,
            profile_identifier="profile",
            vocab_query_config=ExtractionVocabQueryConfig(),
        )

        self.assertEqual(resumed.chunk_results[0].status, "completed")
        self.assertEqual(
            resumed.chunk_results[0].extraction_context.extraction_objects[0].extracted_object.identifier,
            "old",
        )

        stale_overview = persisted_without_overview.model_copy(
            update={
                "initial_extraction_overview": overview_for_file(
                    "sunrise.jpg",
                    summary="sunrise.jpg is present.",
                ),
                "initial_extraction_overview_status": "structured",
            }
        )
        fresh_state = service._prepare_run_state(
            ranking=ranking,
            ordered_chunks=[chunk],
            persisted_state=stale_overview,
            profile_identifier="profile",
            vocab_query_config=ExtractionVocabQueryConfig(),
        )

        self.assertIsNone(fresh_state.initial_extraction_overview)
        self.assertIsNone(fresh_state.initial_extraction_overview_status)
        self.assertEqual(fresh_state.chunk_results[0].status, "pending")
        self.assertIsNone(fresh_state.chunk_results[0].extraction_context)

    async def test_run_complete_workflow_chunks_then_starts_extraction(self):
        service, task_registry, _ = make_service([[make_chunk()]])
        run_calls: list[dict] = []

        async def fake_run_extraction(**kwargs):
            run_calls.append(kwargs)
            return None, TaskStatus.COMPLETED

        service.run_extraction = fake_run_extraction  # type: ignore[method-assign]

        status = await service.run_complete_workflow(
            data_package_id="package-id",
            profile_identifier="profile",
            qualitative_vocab_identifiers=["voc4cat"],
            buffer_window_size=2,
            semantic_chunking_threshold=80.0,
            replace_existing_chunks=True,
            resume=True,
            force_rerun=False,
        )
        await task_registry.wait_for_task("workflow:complete:package-id")

        datasource_service = service.datasource_service
        self.assertEqual(status, TaskStatus.RUNNING)
        self.assertEqual(
            datasource_service.chunk_calls,  # type: ignore[union-attr]
            [
                {
                    "data_package_id": "package-id",
                    "buffer_window_size": 2,
                    "semantic_chunking_threshold": 80.0,
                    "replace_existing_chunks": True,
                }
            ],
        )
        self.assertEqual(
            run_calls,
            [
                {
                    "data_package_id": "package-id",
                    "profile_identifier": "profile",
                    "qualitative_vocab_identifiers": ["voc4cat"],
                    "resume": True,
                }
            ],
        )

    async def test_complete_workflow_can_force_rerun_completed_package(self):
        service, task_registry, output_repository = make_service([[make_chunk()]])
        output_repository.result = ExtractionRunResult(
            generated_final_draft={"id": "old"},
            machine_extraction_context=ExtractionContext(),
            curated_document={"id": "old"},
            draft_quality_state="imperfect_final_draft",
            validation={"status": "valid", "errors": [], "warnings": []},
        )
        run_calls: list[dict] = []

        async def fake_run_extraction(**kwargs):
            run_calls.append(kwargs)
            return None, TaskStatus.COMPLETED

        service.run_extraction = fake_run_extraction  # type: ignore[method-assign]

        async def completed_task():
            return None

        await task_registry.create_task(
            completed_task(),
            type=TaskType.WORKFLOW,
            name="workflow:complete:package-id",
        )
        await task_registry.wait_for_task("workflow:complete:package-id")

        status = await service.run_complete_workflow(
            data_package_id="package-id",
            profile_identifier="profile",
            force_rerun=True,
        )
        await task_registry.wait_for_task("workflow:complete:package-id")

        self.assertEqual(status, TaskStatus.RUNNING)
        self.assertIsNone(output_repository.result)
        self.assertEqual(
            service.datasource_service.chunk_calls[0]["replace_existing_chunks"],  # type: ignore[union-attr]
            False,
        )
        self.assertEqual(len(run_calls), 1)

    async def test_complete_workflow_can_force_rerun_crashed_package(self):
        service, task_registry, output_repository = make_service([[make_chunk()]])
        output_repository.result = ExtractionRunResult(
            generated_final_draft={"id": "stale"},
            machine_extraction_context=ExtractionContext(),
            curated_document={"id": "stale"},
            draft_quality_state="imperfect_final_draft",
            validation={"status": "valid", "errors": [], "warnings": []},
        )
        run_calls: list[dict] = []

        async def crashed_task():
            raise RuntimeError("previous failure")

        await task_registry.create_task(
            crashed_task(),
            type=TaskType.WORKFLOW,
            name="workflow:complete:package-id",
        )
        with self.assertRaises(RuntimeError):
            await task_registry.wait_for_task("workflow:complete:package-id")

        async def fake_run_extraction(**kwargs):
            run_calls.append(kwargs)
            return None, TaskStatus.COMPLETED

        service.run_extraction = fake_run_extraction  # type: ignore[method-assign]

        status = await service.run_complete_workflow(
            data_package_id="package-id",
            profile_identifier="profile",
            force_rerun=True,
        )
        await task_registry.wait_for_task("workflow:complete:package-id")

        self.assertEqual(status, TaskStatus.RUNNING)
        self.assertIsNone(output_repository.result)
        self.assertEqual(len(run_calls), 1)

    async def test_complete_workflow_reports_live_extraction_progress(self):
        service, task_registry, _ = make_service([[make_chunk()]])
        extraction_started = asyncio.Event()
        finish_extraction = asyncio.Event()

        async def fake_run_extraction(**_kwargs):
            extraction_started.set()
            await finish_extraction.wait()
            return None, TaskStatus.COMPLETED

        service.run_extraction = fake_run_extraction  # type: ignore[method-assign]

        status = await service.run_complete_workflow(
            data_package_id="package-id",
            profile_identifier="profile",
        )
        await extraction_started.wait()

        progress_status, progress = await service.get_complete_workflow_progress(
            data_package_id="package-id"
        )
        finish_extraction.set()
        await task_registry.wait_for_task("workflow:complete:package-id")

        self.assertEqual(status, TaskStatus.RUNNING)
        self.assertEqual(progress_status, TaskStatus.RUNNING)
        self.assertIsNotNone(progress)
        self.assertEqual(progress.stage, "extraction")
        self.assertEqual(progress.chunking_status, TaskStatus.COMPLETED)
        self.assertEqual(progress.extraction_status, TaskStatus.RUNNING)
        self.assertEqual(
            {step.name: step.status for step in progress.steps},
            {
                "upload": TaskStatus.COMPLETED,
                "initial_context": TaskStatus.COMPLETED,
                "chunking": TaskStatus.COMPLETED,
                "extraction": TaskStatus.RUNNING,
                "normalization": TaskStatus.UNKNOWN,
                "profile_projection": TaskStatus.UNKNOWN,
                "validation": TaskStatus.UNKNOWN,
            },
        )

    async def test_complete_workflow_progress_recovers_completed_result(self):
        service, _, output_repository = make_service([[make_chunk()]])
        output_repository.result = ExtractionRunResult(
            generated_final_draft={"id": "done"},
            machine_extraction_context=ExtractionContext(),
            curated_document={"id": "done"},
            draft_quality_state="imperfect_final_draft",
            validation={"status": "valid", "errors": [], "warnings": []},
            warnings=["schema-valid but semantically weak"],
        )
        output_repository.run_state = ExtractionRunState(
            profile_identifier="profile",
            chunk_results=[
                ExtractionChunkResult(
                    chunk_index=0,
                    file_path="README.md",
                    start_idx=0,
                    end_idx=0,
                    status="completed",
                    extraction_context=ExtractionContext(),
                )
            ],
        )

        status, progress = await service.get_complete_workflow_progress(
            data_package_id="package-id"
        )

        self.assertEqual(status, TaskStatus.COMPLETED)
        self.assertIsNotNone(progress)
        self.assertEqual(progress.stage, "completed")
        self.assertEqual(progress.profile_identifier, "profile")
        self.assertEqual(progress.result_url, "/api/v1/extraction/result/package-id")
        self.assertEqual(progress.warnings, ["schema-valid but semantically weak"])
        self.assertTrue(all(step.status == TaskStatus.COMPLETED for step in progress.steps))

    async def test_profile_skeleton_creates_minimal_valid_document(self):
        service, _, output_repository = make_service([[make_chunk()]])
        service.profile_service = TitleProfileService()  # type: ignore[assignment]
        context = ExtractionContext.model_validate(
            {
                "extraction_objects": [
                    {
                        "object_kind": "EvaluatedEntity",
                        "extracted_object": {
                            "identifier": "SG-V4050",
                            "description": "Sample SG-V4050",
                            "type": "sample",
                        },
                        "source_text": "##TITLE=SG-V4050",
                    }
                ]
            }
        )
        warnings: list[str] = []

        document = service._fallback_profile_document(
            data_package_id="package-id",
            extraction_context=context,
            validation_schema=TitleProfileService.schema,
        )
        result = await service._save_profile_result(
            data_package_id="package-id",
            profile_identifier="profile",
            extraction_context=context,
            normalization=ExtractionNormalization(),
            document=document,
            profile_manifest=SimpleNamespace(enrichable_fields=[]),
            validation_schema=TitleProfileService.schema,
            state=ExtractionRunState(profile_identifier="profile"),
            warnings=warnings,
        )

        self.assertEqual(result.generated_final_draft, {"title": "SG-V4050", "identifier": "package-id"})
        self.assertEqual(result.curated_document, result.generated_final_draft)
        self.assertEqual(output_repository.result, result)

    async def test_profile_patching_applies_valid_patch_and_persists_interim_document(self):
        service, _, output_repository = make_service([[make_chunk()]])
        context = resource_context("spectrum", "NMR spectrum file.")
        state = ExtractionRunState(profile_identifier="profile")
        progress = ExtractionRunProgress(stage="profile_projection")

        async def fake_generate(*_args, **kwargs):
            if kwargs["output_type"] is ProfilePatchDocument:
                return type_profile_patch("spectrum")
            raise AssertionError("Only profile patch generation is expected")

        with patch("app.services.extraction_service.generate_structured", side_effect=fake_generate):
            document = await service._build_profile_document_by_patching(
                data_package_id="package-id",
                profile_identifier="profile",
                profile_target_class="Dataset",
                extraction_context=context,
                validation_schema=FakeProfileService.schema,
                state=state,
                progress=progress,
                warnings=[],
            )

        self.assertEqual(document["type"], "spectrum")
        self.assertEqual(state.generated_final_draft, document)
        self.assertEqual(state.projection_ledger[0].status, "projected")
        self.assertEqual(output_repository.run_state.generated_final_draft, document)

    async def test_profile_patching_skips_invalid_patch_with_warning(self):
        service, _, _ = make_service([[make_chunk()]])
        context = resource_context("spectrum", "NMR spectrum file.")
        state = ExtractionRunState(profile_identifier="profile")
        progress = ExtractionRunProgress(stage="profile_projection")
        warnings: list[str] = []

        async def fake_generate(*_args, **kwargs):
            if kwargs["output_type"] is ProfilePatchDocument:
                return CompletionResult(
                    output=ProfilePatchDocument(
                        operations=[{"op": "replace", "path": "/id", "value": 12}]
                    ),
                    usage=RunUsage(requests=1),
                )
            raise AssertionError("Only profile patch generation is expected")

        with patch("app.services.extraction_service.generate_structured", side_effect=fake_generate):
            document = await service._build_profile_document_by_patching(
                data_package_id="package-id",
                profile_identifier="profile",
                profile_target_class="Dataset",
                extraction_context=context,
                validation_schema=FakeProfileService.schema,
                state=state,
                progress=progress,
                warnings=warnings,
            )

        self.assertEqual(document["id"], "package-id")
        self.assertEqual(state.projection_ledger[0].status, "user_edit_required")
        self.assertIn("broke schema validation", warnings[0])

    def test_profile_vocab_sources_use_only_quantity_unit_and_enrichable_fields(self):
        sources = ExtractionService._profile_vocab_sources(
            {
                "title": "plain title",
                "type": "dataset",
                "has_quantity_type": "temperature",
                "unit": "K",
            },
            enrichable_fields=["type"],
        )

        self.assertEqual(
            sources,
            [
                ("/type", "type", "dataset"),
                ("/has_quantity_type", "has_quantity_type", "temperature"),
                ("/unit", "unit", "K"),
            ],
        )

    async def test_run_extraction_requires_completed_chunks(self):
        service, _, _ = make_service([])

        with self.assertRaises(ChunkingRequiredError):
            await service.run_extraction(
                data_package_id="package-id",
                profile_identifier="profile",
            )

    async def test_file_ranking_is_deterministic_by_default(self):
        service, _, _ = make_service([[make_chunk()]])

        with patch("app.services.extraction_service.generate_structured") as generate:
            ranking = await service._rank_files(
                data_package_id="package-id",
                data_package=service.datasource_service.get_data_package("package-id"),
                warnings=[],
            )

        generate.assert_not_called()
        self.assertEqual(
            [file.file_path for file in ranking.files],
            ["README.md"],
        )
        self.assertIsNotNone(ranking.files[0].score)
        self.assertTrue(ranking.files[0].reasons)

    async def test_summary_aware_file_ranking_prioritizes_metadata_context(self):
        service, _, _ = make_service([[make_chunk()]])
        data_package = DataPackage(
            file_name="package",
            files=[
                FileEntry(
                    file_path="numbers.txt",
                    file_name="numbers.txt",
                    file_extension=".txt",
                    raw_content=b"1\n2\n3\n4",
                ),
                FileEntry(
                    file_path="dataset_description.txt",
                    file_name="dataset_description.txt",
                    file_extension=".txt",
                    raw_content=b"dataset description",
                ),
                FileEntry(
                    file_path="acquisition.txt",
                    file_name="acquisition.txt",
                    file_extension=".txt",
                    raw_content=b"acquisition settings",
                ),
            ],
        )
        state = ExtractionRunState(
            initial_file_summaries=[
                ExtractionFileSummary(
                    file_path="numbers.txt",
                    data_format="plain text",
                    quantitative_signals=["single column numeric values"],
                ),
                ExtractionFileSummary(
                    file_path="dataset_description.txt",
                    explicit_purpose="dataset description",
                    purpose_evidence=["dataset description"],
                    metadata_signals=["instrument: Raman microscope"],
                ),
                ExtractionFileSummary(
                    file_path="acquisition.txt",
                    metadata_signals=["acquisition settings"],
                    instrument_or_software_terms_and_settings=["instrument settings"],
                ),
            ],
        )

        ranking = service._rank_files_from_summaries(
            data_package=data_package,
            state=state,
        )

        self.assertEqual(
            [file.file_path for file in ranking.files],
            ["dataset_description.txt", "acquisition.txt", "numbers.txt"],
        )
        self.assertGreater(ranking.files[0].score or 0, ranking.files[-1].score or 0)
        self.assertIn("explicit dataset/package documentation", ranking.files[0].reasons)
        self.assertTrue(all(file.reasons for file in ranking.files))

    async def test_run_extraction_processes_task_and_persists_interim_context(self):
        service, task_registry, output_repository = make_service(
            [[make_chunk(0, "sample one"), make_chunk(1, "sample two")]]
        )
        outputs = [
            CompletionResult(
                output=resource_context("alpha-resource", "Alpha catalyst metadata."),
                usage=RunUsage(requests=1, input_tokens=20, output_tokens=5),
            ),
            CompletionResult(
                output=resource_context("beta-spectrum", "Beta NMR spectrum file."),
                usage=RunUsage(requests=1, input_tokens=20, output_tokens=5),
            ),
            CompletionResult(
                output={"id": "dataset"},
                usage=RunUsage(requests=1, input_tokens=30, output_tokens=8),
            ),
        ]

        async def fake_generate(*_args, **_kwargs):
            if _kwargs["output_type"] is ProfilePatchDocument:
                return empty_profile_patch()
            return outputs.pop(0)

        with patch("app.services.extraction_service.generate_structured", side_effect=fake_generate):
            result, status = await service.run_extraction(
                data_package_id="package-id",
                profile_identifier="profile",
            )
            self.assertIsNone(result)
            self.assertEqual(status, TaskStatus.RUNNING)
            await task_registry.wait_for_task("extraction:run:package-id", timeout=2)

        self.assertIsNotNone(output_repository.context)
        self.assertGreaterEqual(len(output_repository.contexts), 2)
        self.assertEqual(output_repository.contexts[0].resources[0].identifier, "alpha-resource")
        self.assertEqual(len(output_repository.contexts[1].resources), 2)
        self.assertIsNotNone(output_repository.result)
        self.assertEqual(output_repository.result.generated_final_draft["id"], "package-id")
        self.assertEqual(len(output_repository.run_state.projection_ledger), 3)
        self.assertIn("chunk_extraction", output_repository.token_usage)

    async def test_context_target_stops_after_interim_context_without_result(self):
        service, task_registry, output_repository = make_service([[make_chunk()]])

        async def fake_generate(*_args, **kwargs):
            self.assertIs(kwargs["output_type"], EvidenceContext)
            return CompletionResult(
                output=EvidenceContext(
                    notes=[
                        EvidenceNote(
                            note_id="context-only",
                            category="resource_signal",
                            observation="Context only resource.",
                            evidence_text="metadata",
                            signal_level="high",
                        )
                    ]
                ),
                usage=RunUsage(requests=1),
            )

        with patch("app.services.extraction_service.generate_structured", side_effect=fake_generate):
            result, status = await service.run_extraction(
                data_package_id="package-id",
                target_stage="context",
            )
            self.assertIsNone(result)
            self.assertEqual(status, TaskStatus.RUNNING)
            await task_registry.wait_for_task("extraction:run:package-id", timeout=2)

        self.assertIsNotNone(output_repository.evidence_context)
        self.assertIsNone(output_repository.result)
        self.assertIsNone(output_repository.run_state.profile_identifier)
        self.assertIsNone(output_repository.run_state.generated_final_draft)

        status, progress = await service.get_extraction_progress(data_package_id="package-id")
        self.assertEqual(status, TaskStatus.COMPLETED)
        self.assertIsNotNone(progress)
        self.assertEqual(progress.stage, "interim_evidence_context")

    async def test_profile_target_requires_selected_profile(self):
        service, _, _ = make_service([[make_chunk()]])

        with self.assertRaisesRegex(
            ValueError,
            "profile identifier is required",
        ):
            await service.run_extraction(
                data_package_id="package-id",
                target_stage="profile",
            )

    async def test_profile_target_resumes_context_and_stops_after_profile_draft(self):
        service, task_registry, output_repository = make_service([[make_chunk()]])
        output_repository.save_extraction_run_state(
            workflow_id="package-id",
            state=ExtractionRunState(
                profile_identifier="profile",
                ranked_files=[RankedFile(rank=1, file_path="README.md")],
                initial_file_summaries=[
                    ExtractionFileSummary(
                        file_path="README.md",
                        data_format="plain text",
                    )
                ],
                initial_file_summary_status="completed",
                initial_extraction_overview=overview_for_file(
                    "README.md",
                    summary="README.md is present.",
                ),
                initial_extraction_overview_status="structured",
                chunk_results=[
                    ExtractionChunkResult(
                        chunk_index=0,
                        file_path="README.md",
                        start_idx=0,
                        end_idx=0,
                        status="completed",
                        extraction_context=resource_context("profile-resource", "Profile resource."),
                    ),
                ],
            ),
        )
        output_repository.save_extraction_context(
            workflow_id="package-id",
            extraction_context=resource_context("profile-resource", "Profile resource."),
        )

        async def fake_generate(*_args, **kwargs):
            self.assertIs(kwargs["output_type"], ProfilePatchDocument)
            return type_profile_patch()

        with patch("app.services.extraction_service.generate_structured", side_effect=fake_generate):
            result, status = await service.run_extraction(
                data_package_id="package-id",
                profile_identifier="profile",
                resume=True,
                target_stage="profile",
            )
            self.assertIsNone(result)
            self.assertEqual(status, TaskStatus.RUNNING)
            await task_registry.wait_for_task("extraction:run:package-id", timeout=2)

        self.assertIsNone(output_repository.result)
        self.assertIsNotNone(output_repository.run_state.generated_final_draft)
        self.assertEqual(output_repository.run_state.generated_final_draft["type"], "dataset")
        self.assertEqual(output_repository.run_state.vocab_queries, [])
        status, progress = await service.get_extraction_progress(data_package_id="package-id")
        self.assertEqual(status, TaskStatus.COMPLETED)
        self.assertIsNotNone(progress)
        self.assertEqual(progress.stage, "profile_draft")

    async def test_grounding_target_uses_manual_interim_profile_and_writes_result(self):
        service, task_registry, output_repository = make_service([[make_chunk()]])
        service.semantic_service = FakeSemanticService()  # type: ignore[assignment]
        output_repository.save_extraction_run_state(
            workflow_id="package-id",
            state=ExtractionRunState(
                profile_identifier="profile",
                ranked_files=[RankedFile(rank=1, file_path="README.md")],
                initial_file_summaries=[
                    ExtractionFileSummary(
                        file_path="README.md",
                        data_format="plain text",
                    )
                ],
                initial_file_summary_status="completed",
                initial_extraction_overview=overview_for_file(
                    "README.md",
                    summary="README.md is present.",
                ),
                initial_extraction_overview_status="structured",
                chunk_results=[
                    ExtractionChunkResult(
                        chunk_index=0,
                        file_path="README.md",
                        start_idx=0,
                        end_idx=0,
                        status="completed",
                        extraction_context=resource_context("grounding-resource", "Grounding resource."),
                    ),
                ],
                generated_final_draft={"id": "manual-id", "type": "dataset"},
                curated_document={"id": "manual-id", "type": "dataset"},
            ),
        )
        output_repository.save_extraction_context(
            workflow_id="package-id",
            extraction_context=resource_context("grounding-resource", "Grounding resource."),
        )

        async def fake_generate(*_args, **kwargs):
            output_type = kwargs["output_type"]
            if output_type is ProfilePatchDocument:
                return empty_profile_patch()
            return CompletionResult(
                output=VocabularyCandidateSelection(
                    selected_uri=None,
                    confidence=0.0,
                    reason="No clear mapping.",
                ),
                usage=RunUsage(requests=1),
            )

        with patch("app.services.extraction_service.generate_structured", side_effect=fake_generate):
            result, status = await service.run_extraction(
                data_package_id="package-id",
                profile_identifier="profile",
                resume=True,
                target_stage="grounding",
            )
            self.assertIsNone(result)
            self.assertEqual(status, TaskStatus.RUNNING)
            await task_registry.wait_for_task("extraction:run:package-id", timeout=2)

        self.assertIsNotNone(output_repository.result)
        self.assertEqual(output_repository.result.generated_final_draft["id"], "manual-id")
        self.assertEqual(output_repository.result.curated_document["id"], "manual-id")
        self.assertGreaterEqual(len(output_repository.run_state.vocab_queries), 1)

    async def test_update_curated_document_saves_invalid_edits_without_changing_generated_draft(self):
        service, _, output_repository = make_service([[make_chunk()]])
        output_repository.save_extraction_run_state(
            workflow_id="package-id",
            state=ExtractionRunState(
                profile_identifier="profile",
                chunk_results=[
                    ExtractionChunkResult(
                        chunk_index=0,
                        file_path="README.md",
                        start_idx=0,
                        end_idx=0,
                        status="completed",
                        extraction_context=resource_context("resource", "Persisted result."),
                    ),
                ],
                generated_final_draft={"id": "generated", "type": "dataset"},
                curated_document={"id": "previous"},
            ),
        )

        progress = await service.update_curated_document(
            data_package_id="package-id",
            profile_identifier="profile",
            document={"id": "edited", "type": "dataset"},
        )

        self.assertEqual(progress.stage, "curated_document")
        self.assertEqual(output_repository.run_state.generated_final_draft["id"], "generated")
        self.assertEqual(output_repository.run_state.curated_document["id"], "edited")

        invalid_progress = await service.update_curated_document(
            data_package_id="package-id",
            profile_identifier="profile",
            document={"id": None},
        )

        self.assertEqual(output_repository.run_state.generated_final_draft["id"], "generated")
        self.assertNotIn("id", output_repository.run_state.curated_document)
        self.assertEqual(invalid_progress.curated_validation.status, "invalid")

    async def test_progress_returns_persisted_interim_context(self):
        service, _, output_repository = make_service([[make_chunk()]])
        output_repository.save_extraction_context(
            workflow_id="package-id",
            extraction_context=resource_context("interim-dataset", "Persisted partial context."),
        )

        status, progress = await service.get_extraction_progress(
            data_package_id="package-id",
        )

        self.assertEqual(status, TaskStatus.UNKNOWN)
        self.assertIsNotNone(progress)
        self.assertEqual(progress.stage, "interim_context")
        self.assertIsNotNone(progress.interim_context)
        self.assertEqual(
            progress.interim_context.resources[0].identifier,
            "interim-dataset",
        )

    async def test_chunk_repairs_run_after_first_pass_extraction_calls(self):
        service, task_registry, output_repository = make_service(
            [[make_chunk(0, "sample one"), make_chunk(1, "sample two")]]
        )
        call_order: list[str] = []
        chunk_outputs = [
            CompletionResult(
                output=EvidenceContext(
                    notes=[
                        EvidenceNote(
                            note_id="resource-one",
                            category="resource_signal",
                            observation="First partial resource.",
                            evidence_text="sample one",
                            signal_level="high",
                        )
                    ]
                ),
                usage=RunUsage(requests=1, input_tokens=20, output_tokens=5),
            ),
            MaxRetriesExceeded(
                last_error=OutputParsingError("bad json"),
                failed_response='{"extraction_objects": [',
                usage=RunUsage(requests=1, input_tokens=21, output_tokens=4),
            ),
        ]

        async def fake_generate(*_args, **kwargs):
            output_type = kwargs["output_type"]
            if output_type is EvidenceContext:
                call_order.append(f"extract:{len(call_order)}")
                output = chunk_outputs.pop(0)
                if isinstance(output, Exception):
                    raise output
                return output
            if output_type is ProfilePatchDocument:
                call_order.append("profile_patch")
                return empty_profile_patch()
            call_order.append("profile")
            return CompletionResult(
                output={"id": "dataset"},
                usage=RunUsage(requests=1, input_tokens=30, output_tokens=8),
            )

        async def fake_repair(*_args, **_kwargs):
            call_order.append("repair")
            return CompletionResult(
                output=EvidenceContext(
                    notes=[
                        EvidenceNote(
                            note_id="resource-two",
                            category="resource_signal",
                            observation="Repaired resource.",
                            evidence_text="sample two",
                            signal_level="high",
                        )
                    ]
                ),
                usage=RunUsage(requests=1, input_tokens=12, output_tokens=3),
            )

        with (
            patch("app.services.extraction_service.generate_structured", side_effect=fake_generate),
            patch("app.services.extraction_service.repair_structured_output", side_effect=fake_repair),
        ):
            result, status = await service.run_extraction(
                data_package_id="package-id",
                profile_identifier="profile",
            )
            self.assertIsNone(result)
            self.assertEqual(status, TaskStatus.RUNNING)
            await task_registry.wait_for_task("extraction:run:package-id", timeout=2)

        self.assertEqual(call_order[:3], ["extract:0", "extract:1", "repair"])
        self.assertIsNotNone(output_repository.run_state)
        self.assertEqual(
            [chunk.status for chunk in output_repository.run_state.chunk_results],
            ["completed", "completed"],
        )
        self.assertIsNotNone(output_repository.evidence_context)
        self.assertEqual(
            [note.note_id for note in output_repository.evidence_context.notes],
            ["resource-one", "resource-two"],
        )
        self.assertIn("chunk_extraction_repair", output_repository.token_usage)

    async def test_failed_chunk_repair_does_not_crash_extraction(self):
        service, task_registry, output_repository = make_service(
            [[make_chunk(0, "sample one"), make_chunk(1, "sample two")]]
        )
        chunk_outputs = [
            CompletionResult(
                output=EvidenceContext(
                    notes=[
                        EvidenceNote(
                            note_id="resource-one",
                            category="resource_signal",
                            observation="First partial resource.",
                            evidence_text="sample one",
                            signal_level="high",
                        )
                    ]
                ),
                usage=RunUsage(requests=1, input_tokens=20, output_tokens=5),
            ),
            MaxRetriesExceeded(
                last_error=OutputParsingError("bad json"),
                failed_response='{"extraction_objects": [',
                usage=RunUsage(requests=1, input_tokens=21, output_tokens=4),
            ),
        ]

        async def fake_generate(*_args, **kwargs):
            output_type = kwargs["output_type"]
            if output_type is EvidenceContext:
                output = chunk_outputs.pop(0)
                if isinstance(output, Exception):
                    raise output
                return output
            if output_type is ProfilePatchDocument:
                return empty_profile_patch()
            return CompletionResult(
                output={"id": "dataset"},
                usage=RunUsage(requests=1, input_tokens=30, output_tokens=8),
            )

        async def fake_repair(*_args, **_kwargs):
            raise MaxRetriesExceeded(
                last_error=OutputParsingError("still bad"),
                failed_response="still bad",
                usage=RunUsage(requests=1, input_tokens=12, output_tokens=3),
            )

        with (
            patch("app.services.extraction_service.generate_structured", side_effect=fake_generate),
            patch("app.services.extraction_service.repair_structured_output", side_effect=fake_repair),
        ):
            result, status = await service.run_extraction(
                data_package_id="package-id",
                profile_identifier="profile",
            )
            self.assertIsNone(result)
            self.assertEqual(status, TaskStatus.RUNNING)
            await task_registry.wait_for_task("extraction:run:package-id", timeout=2)

        self.assertIsNotNone(output_repository.result)
        self.assertEqual(output_repository.result.generated_final_draft["id"], "package-id")
        self.assertEqual(output_repository.run_state.chunk_results[0].status, "completed")
        self.assertEqual(output_repository.run_state.chunk_results[1].status, "failed")
        self.assertIn("Chunk extraction repair failed", output_repository.warnings[0])
        self.assertIn("chunk_extraction_repair", output_repository.token_usage)

    async def test_immediate_chunk_repair_runs_before_next_chunk(self):
        service, task_registry, output_repository = make_service(
            [[make_chunk(0, "sample one"), make_chunk(1, "sample two")]]
        )
        call_order: list[str] = []
        chunk_outputs = [
            MaxRetriesExceeded(
                last_error=OutputParsingError("bad json"),
                failed_response='{"notes": [',
                usage=RunUsage(requests=1, input_tokens=21, output_tokens=4),
            ),
            CompletionResult(
                output=EvidenceContext(
                    notes=[
                        EvidenceNote(
                            note_id="resource-two",
                            category="resource_signal",
                            observation="Second chunk evidence.",
                            evidence_text="sample two",
                            signal_level="high",
                        )
                    ]
                ),
                usage=RunUsage(requests=1, input_tokens=20, output_tokens=5),
            ),
        ]

        async def fake_generate(*_args, **kwargs):
            if kwargs["output_type"] is EvidenceContext:
                call_order.append(f"extract:{sum(1 for item in call_order if item.startswith('extract'))}")
                output = chunk_outputs.pop(0)
                if isinstance(output, Exception):
                    raise output
                return output
            return CompletionResult(output={"id": "dataset"}, usage=RunUsage(requests=1))

        async def fake_repair(*_args, **_kwargs):
            call_order.append("repair")
            return CompletionResult(
                output=EvidenceContext(
                    notes=[
                        EvidenceNote(
                            note_id="resource-one",
                            category="resource_signal",
                            observation="Repaired first chunk evidence.",
                            evidence_text="sample one",
                            signal_level="high",
                        )
                    ]
                ),
                usage=RunUsage(requests=1, input_tokens=12, output_tokens=3),
            )

        with (
            patch("app.services.extraction_service.generate_structured", side_effect=fake_generate),
            patch("app.services.extraction_service.repair_structured_output", side_effect=fake_repair),
        ):
            result, status = await service.run_extraction(
                data_package_id="package-id",
                profile_identifier="profile",
                target_stage="context",
                chunk_repair_mode="immediate",
            )
            self.assertIsNone(result)
            self.assertEqual(status, TaskStatus.RUNNING)
            await task_registry.wait_for_task("extraction:run:package-id", timeout=2)

        self.assertEqual(call_order, ["extract:0", "repair", "extract:1"])
        self.assertEqual(output_repository.run_state.chunk_repair_mode, "immediate")
        self.assertEqual(
            [chunk.status for chunk in output_repository.run_state.chunk_results],
            ["completed", "completed"],
        )

    async def test_context_extraction_uses_configured_prompt_tokenizer_budget(self):
        class FakeEncoding:
            def __init__(self, text: str):
                parts = text.split()
                self.ids = list(range(len(parts)))
                offset = 0
                offsets = []
                for part in parts:
                    start = text.find(part, offset)
                    end = start + len(part)
                    offsets.append((start, end))
                    offset = end
                self.offsets = offsets

        class FakeTokenizer:
            def encode(self, text: str, add_special_tokens: bool = False):
                return FakeEncoding(text)

        service, task_registry, output_repository = make_service(
            [[make_chunk(0, "sample one")]]
        )
        service.settings.ollama_chat_tokenizer = "example/tokenizer"
        assert output_repository.run_state is not None
        output_repository.run_state.initial_extraction_overview = overview_for_file(
            "README.md",
            source_fingerprint="overview",
            summary="signal " + "word " * 500,
        )
        budgeter = PromptTokenBudgeter(tokenizer=FakeTokenizer())
        captured_system_prompts: list[str] = []

        async def fake_generate(*_args, **kwargs):
            if kwargs["output_type"] is EvidenceContext:
                captured_system_prompts.append(kwargs["system"])
                return CompletionResult(
                    output=EvidenceContext(
                        notes=[
                            EvidenceNote(
                                note_id="resource-one",
                                category="resource_signal",
                                observation="First chunk evidence.",
                                evidence_text="sample one",
                                signal_level="high",
                            )
                        ]
                    ),
                    usage=RunUsage(requests=1, input_tokens=20, output_tokens=5),
                )
            return CompletionResult(output={"id": "dataset"}, usage=RunUsage(requests=1))

        with (
            patch("app.services.extraction_service.generate_structured", side_effect=fake_generate),
            patch(
                "app.services.extraction_service.PromptTokenBudgeter.from_tokenizer_source",
                return_value=budgeter,
            ) as tokenizer_loader,
        ):
            result, status = await service.run_extraction(
                data_package_id="package-id",
                profile_identifier="profile",
                target_stage="context",
            )
            self.assertIsNone(result)
            self.assertEqual(status, TaskStatus.RUNNING)
            await task_registry.wait_for_task("extraction:run:package-id", timeout=2)

        tokenizer_loader.assert_called_once_with("example/tokenizer")
        self.assertEqual(len(captured_system_prompts), 1)
        self.assertIn("[orientation truncated]", captured_system_prompts[0])
        self.assertFalse(
            any(
                "conservative estimates" in warning
                for warning in output_repository.warnings
            )
        )

    async def test_disabled_chunk_repair_leaves_first_pass_failure_failed(self):
        service, task_registry, output_repository = make_service(
            [[make_chunk(0, "sample one"), make_chunk(1, "sample two")]]
        )
        repair_called = False
        chunk_outputs = [
            MaxRetriesExceeded(
                last_error=OutputParsingError("bad json"),
                failed_response='{"notes": [',
                usage=RunUsage(requests=1, input_tokens=21, output_tokens=4),
            ),
            CompletionResult(
                output=EvidenceContext(
                    notes=[
                        EvidenceNote(
                            note_id="resource-two",
                            category="resource_signal",
                            observation="Second chunk evidence.",
                            evidence_text="sample two",
                            signal_level="high",
                        )
                    ]
                ),
                usage=RunUsage(requests=1, input_tokens=20, output_tokens=5),
            ),
        ]

        async def fake_generate(*_args, **kwargs):
            if kwargs["output_type"] is EvidenceContext:
                output = chunk_outputs.pop(0)
                if isinstance(output, Exception):
                    raise output
                return output
            return CompletionResult(output={"id": "dataset"}, usage=RunUsage(requests=1))

        async def fake_repair(*_args, **_kwargs):
            nonlocal repair_called
            repair_called = True
            return CompletionResult(output=EvidenceContext(), usage=RunUsage(requests=1))

        with (
            patch("app.services.extraction_service.generate_structured", side_effect=fake_generate),
            patch("app.services.extraction_service.repair_structured_output", side_effect=fake_repair),
        ):
            result, status = await service.run_extraction(
                data_package_id="package-id",
                profile_identifier="profile",
                target_stage="context",
                chunk_repair_mode="disabled",
            )
            self.assertIsNone(result)
            self.assertEqual(status, TaskStatus.RUNNING)
            await task_registry.wait_for_task("extraction:run:package-id", timeout=2)

        self.assertFalse(repair_called)
        self.assertEqual(output_repository.run_state.chunk_repair_mode, "disabled")
        self.assertEqual(
            [chunk.status for chunk in output_repository.run_state.chunk_results],
            ["failed", "completed"],
        )

    async def test_resume_reuses_completed_chunk_results(self):
        service, task_registry, output_repository = make_service(
            [[make_chunk(0, "sample one"), make_chunk(1, "sample two")]]
        )
        output_repository.save_extraction_run_state(
            workflow_id="package-id",
            state=ExtractionRunState(
                ranked_files=[RankedFile(rank=1, file_path="README.md")],
                initial_file_summaries=[
                    ExtractionFileSummary(
                        file_path="README.md",
                        data_format="plain text",
                    )
                ],
                initial_file_summary_status="completed",
                initial_extraction_overview=overview_for_file(
                    "README.md",
                    summary="README.md is present.",
                ),
                initial_extraction_overview_status="structured",
                chunk_results=[
                    ExtractionChunkResult(
                        chunk_index=0,
                        file_path="README.md",
                        start_idx=0,
                        end_idx=0,
                        status="completed",
                        extraction_context=resource_context(
                            "already-extracted",
                            "Persisted alpha catalyst result.",
                        ),
                    ),
                    ExtractionChunkResult(
                        chunk_index=1,
                        file_path="README.md",
                        start_idx=1,
                        end_idx=1,
                        status="pending",
                    ),
                ],
            ),
        )
        outputs = [
            CompletionResult(
                output=resource_context("resumed-chunk", "NMR spectrum details."),
                usage=RunUsage(requests=1, input_tokens=20, output_tokens=5),
            ),
            CompletionResult(
                output={"id": "dataset"},
                usage=RunUsage(requests=1, input_tokens=30, output_tokens=8),
            ),
        ]

        async def fake_generate(*_args, **_kwargs):
            if _kwargs["output_type"] is ProfilePatchDocument:
                return empty_profile_patch()
            return outputs.pop(0)

        with patch("app.services.extraction_service.generate_structured", side_effect=fake_generate):
            result, status = await service.run_extraction(
                data_package_id="package-id",
                profile_identifier="profile",
                resume=True,
            )
            self.assertIsNone(result)
            self.assertEqual(status, TaskStatus.RUNNING)
            await task_registry.wait_for_task("extraction:run:package-id", timeout=2)

        self.assertEqual(len(outputs), 1)
        self.assertIsNotNone(output_repository.result)
        self.assertIsNotNone(output_repository.run_state)
        self.assertEqual(
            [chunk.status for chunk in output_repository.run_state.chunk_results],
            ["completed", "completed"],
        )
        self.assertEqual(
            [resource.identifier for resource in output_repository.context.resources],
            ["already-extracted", "resumed-chunk", "README.md"],
        )

    async def test_vocab_candidate_discovery_starts_after_all_chunks_finish(self):
        service, task_registry, output_repository = make_service(
            [[make_chunk(0, "sample one"), make_chunk(1, "sample two")]]
        )
        semantic_service = FakeSemanticService()
        service.semantic_service = semantic_service  # type: ignore[assignment]
        service.settings.extraction_vocab_query_concurrency = 1
        call_order: list[str] = []

        async def fake_generate(*_args, **kwargs):
            output_type = kwargs["output_type"]
            if output_type is ExtractionContext:
                if len([item for item in call_order if item.startswith("extract")]) == 1:
                    self.assertEqual(semantic_service.query_calls, [])
                call_order.append("extract")
                return CompletionResult(
                    output=qualitative_context("sample", "phase", "liquid"),
                    usage=RunUsage(requests=1),
                )
            if output_type is ProfilePatchDocument:
                call_order.append("profile_patch")
                return type_profile_patch()
            call_order.append("profile")
            return CompletionResult(output={"id": "dataset"}, usage=RunUsage(requests=1))

        with patch("app.services.extraction_service.generate_structured", side_effect=fake_generate):
            await service.run_extraction(
                data_package_id="package-id",
                profile_identifier="profile",
            )
            await task_registry.wait_for_task("extraction:run:package-id", timeout=2)

        self.assertGreaterEqual(len(semantic_service.query_calls), 1)
        self.assertIsNotNone(output_repository.run_state)
        vocab_queries = output_repository.run_state.vocab_queries
        self.assertGreaterEqual(len(vocab_queries), 1)
        self.assertEqual(vocab_queries[0].status, "completed")
        self.assertIsNotNone(vocab_queries[0].query)
        self.assertIsNotNone(vocab_queries[0].result)

    async def test_profile_type_field_gets_voc4cat_query_after_patching(self):
        service, task_registry, output_repository = make_service(
            [[make_chunk(0, "first"), make_chunk(1, "second")]]
        )
        semantic_service = FakeSemanticService()
        service.semantic_service = semantic_service  # type: ignore[assignment]
        contexts = [
            resource_context("resource-a", "First resource."),
            resource_context("resource-b", "Second resource."),
        ]

        async def fake_generate(*_args, **kwargs):
            output_type = kwargs["output_type"]
            if output_type is ExtractionContext:
                return CompletionResult(
                    output=contexts.pop(0),
                    usage=RunUsage(requests=1),
                )
            if output_type is ProfilePatchDocument:
                return type_profile_patch()
            return CompletionResult(output={"id": "dataset"}, usage=RunUsage(requests=1))

        with patch("app.services.extraction_service.generate_structured", side_effect=fake_generate):
            await service.run_extraction(
                data_package_id="package-id",
                profile_identifier="profile",
            )
            await task_registry.wait_for_task("extraction:run:package-id", timeout=2)

        self.assertIsNotNone(output_repository.run_state)
        object_queries = [
            record
            for record in output_repository.run_state.vocab_queries
            if record.kind == "profile_type"
        ]
        self.assertEqual(len(object_queries), 1)
        self.assertEqual(
            [record.source_context["json_path"] for record in object_queries],
            ["/type"],
        )

    async def test_profile_quantity_fields_are_queried_once_after_patching(self):
        service, task_registry, output_repository = make_service(
            [[make_chunk(0, "first"), make_chunk(1, "second")]]
        )
        semantic_service = FakeSemanticService()
        service.semantic_service = semantic_service  # type: ignore[assignment]

        async def fake_generate(*_args, **kwargs):
            output_type = kwargs["output_type"]
            if output_type is ExtractionContext:
                return CompletionResult(
                    output=quantitative_context("same-resource"),
                    usage=RunUsage(requests=1),
                )
            if output_type is ProfilePatchDocument:
                return quantity_profile_patch()
            if output_type is VocabularyFallbackQuery:
                return CompletionResult(
                    output=VocabularyFallbackQuery(
                        vector_query="temperature",
                        fulltext_query="temperature",
                    ),
                    usage=RunUsage(requests=1),
                )
            return CompletionResult(output={"id": "dataset"}, usage=RunUsage(requests=1))

        with patch("app.services.extraction_service.generate_structured", side_effect=fake_generate):
            await service.run_extraction(
                data_package_id="package-id",
                profile_identifier="profile",
            )
            await task_registry.wait_for_task("extraction:run:package-id", timeout=2)

        self.assertIsNotNone(output_repository.run_state)
        records = output_repository.run_state.vocab_queries
        quantity_records = [
            record
            for record in records
            if record.kind in {"profile_has_quantity_type", "profile_unit"}
        ]
        object_records = [
            record for record in records if record.kind == "profile_type"
        ]
        self.assertEqual(
            [record.kind for record in quantity_records],
            ["profile_has_quantity_type", "profile_unit"],
        )
        self.assertEqual(len(object_records), 0)

    async def test_vocab_query_config_update_persists_in_run_state_and_progress(self):
        service, _, output_repository = make_service([[make_chunk()]])
        output_repository.save_extraction_run_state(
            workflow_id="package-id",
            state=ExtractionRunState(
                profile_identifier="profile",
                chunk_results=[
                    ExtractionChunkResult(
                        chunk_index=0,
                        file_path="README.md",
                        start_idx=0,
                        end_idx=0,
                        status="completed",
                        extraction_context=resource_context("resource", "Persisted result."),
                    )
                ],
            ),
        )

        updated_config = output_repository.run_state.vocab_query_config.model_copy(
            update={"vector_top_k": 3, "fulltext_top_k": 4}
        )

        progress = await service.update_vocab_query_config(
            data_package_id="package-id",
            config=updated_config,
        )

        self.assertEqual(output_repository.run_state.vocab_query_config.vector_top_k, 3)
        self.assertEqual(progress.vocab_query_config.fulltext_top_k, 4)

    async def test_quantitative_vocab_query_config_is_independent_from_qualitative_config(self):
        config = ExtractionVocabQueryConfig(
            vector_top_k=3,
            fulltext_top_k=4,
            seed_top_k=5,
            quantitative_vector_top_k=13,
            quantitative_fulltext_top_k=14,
            quantitative_seed_top_k=15,
            quantitative_max_hops=2,
            quantitative_max_statements_per_seed=70,
            quantitative_traversal_direction="outgoing",
            quantitative_vector_weight=1.5,
            quantitative_fulltext_weight=0.5,
            quantitative_rrf_k=80,
        )
        query = VocabQuery(
            rdf_type="qudt__QuantityKind",
            vector_query="temperature",
            fulltext_query="temperature",
        )

        qualitative_query = ExtractionService._configured_vocab_query(query, config)
        quantitative_query = ExtractionService._configured_vocab_query(query, config, group="quantitative")

        self.assertEqual(qualitative_query.vector_top_k, 3)
        self.assertEqual(qualitative_query.fulltext_top_k, 4)
        self.assertEqual(qualitative_query.seed_top_k, 5)
        self.assertEqual(quantitative_query.vector_top_k, 13)
        self.assertEqual(quantitative_query.fulltext_top_k, 14)
        self.assertEqual(quantitative_query.seed_top_k, 15)
        self.assertEqual(quantitative_query.max_hops, 2)
        self.assertEqual(quantitative_query.max_statements_per_seed, 70)
        self.assertEqual(quantitative_query.traversal_direction, "outgoing")
        self.assertEqual(quantitative_query.vector_weight, 1.5)
        self.assertEqual(quantitative_query.fulltext_weight, 0.5)
        self.assertEqual(quantitative_query.rrf_k, 80)

    async def test_vocab_selection_is_serial_in_conservative_mode(self):
        service, _, _ = make_service([[make_chunk()]])
        service.settings.vocab_selection_parallel_mode = "conservative"
        service.settings.vocab_selection_llm_concurrency = 2
        max_active = 0
        active = 0
        state = ExtractionRunState()

        async def completed_discovery(index: int):
            query_id = f"query-{index}"
            state.vocab_queries.append(
                ExtractionVocabQueryRecord(
                    query_id=query_id,
                    kind="qualitative_attribute",
                    source_value="phase: liquid",
                    source_context={"title": "phase", "value": "liquid"},
                    vocabulary_identifier="urn:vocab",
                    rdf_type="skos__Concept",
                    query=VocabQuery(rdf_type="skos__Concept", fulltext_query="phase liquid"),
                    status="completed",
                    result=VocabQueryResult(
                        identifier="urn:vocab",
                        rdf_type="skos__Concept",
                        resources={
                            f"urn:candidate:{index}": CompactVocabResource(
                                uri=f"urn:candidate:{index}",
                                rdf_types=["skos__Concept"],
                                properties={"label": f"candidate {index}"},
                            )
                        },
                    ),
                )
            )
            return _QualitativeCandidateDiscovery(
                attribute=qualitative_context(f"sample-{index}", "phase", "liquid")
                .resources[0]
                .has_qualitative_attributes[0],
                query_ids=[query_id],
            )

        async def fake_generate(*_args, **kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return CompletionResult(
                output=VocabularyCandidateSelection(selected_uri=None, confidence=0.0),
                usage=RunUsage(requests=1),
            )

        tasks = [asyncio.create_task(completed_discovery(index)) for index in range(3)]
        extraction_context = ExtractionContext()
        with patch("app.services.extraction_service.generate_structured", side_effect=fake_generate):
            normalization = await service._normalize_from_candidate_tasks(
                data_package_id="package-id",
                state=state,
                candidate_tasks=tasks,
                extraction_context=extraction_context,
                warnings=[],
            )

        self.assertEqual(max_active, 1)
        self.assertEqual(len(normalization.qualitative_attributes), 3)

    async def test_vocab_selection_parallel_mode_allows_overlapping_llm_calls(self):
        service, _, _ = make_service([[make_chunk()]])
        service.settings.vocab_selection_parallel_mode = "parallel"
        service.settings.vocab_selection_llm_concurrency = 2
        max_active = 0
        active = 0
        state = ExtractionRunState()

        async def completed_discovery(index: int):
            query_id = f"query-{index}"
            state.vocab_queries.append(
                ExtractionVocabQueryRecord(
                    query_id=query_id,
                    kind="qualitative_attribute",
                    source_value="phase: liquid",
                    source_context={"title": "phase", "value": "liquid"},
                    vocabulary_identifier="urn:vocab",
                    rdf_type="skos__Concept",
                    query=VocabQuery(rdf_type="skos__Concept", fulltext_query="phase liquid"),
                    status="completed",
                    result=VocabQueryResult(
                        identifier="urn:vocab",
                        rdf_type="skos__Concept",
                        resources={
                            f"urn:candidate:{index}": CompactVocabResource(
                                uri=f"urn:candidate:{index}",
                                rdf_types=["skos__Concept"],
                                properties={"label": f"candidate {index}"},
                            )
                        },
                    ),
                )
            )
            return _QualitativeCandidateDiscovery(
                attribute=qualitative_context(f"sample-{index}", "phase", "liquid")
                .resources[0]
                .has_qualitative_attributes[0],
                query_ids=[query_id],
            )

        async def fake_generate(*_args, **_kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.03)
            active -= 1
            return CompletionResult(
                output=VocabularyCandidateSelection(selected_uri=None, confidence=0.0),
                usage=RunUsage(requests=1),
            )

        tasks = [asyncio.create_task(completed_discovery(index)) for index in range(3)]
        extraction_context = ExtractionContext()
        with patch("app.services.extraction_service.generate_structured", side_effect=fake_generate):
            await service._normalize_from_candidate_tasks(
                data_package_id="package-id",
                state=state,
                candidate_tasks=tasks,
                extraction_context=extraction_context,
                warnings=[],
            )

        self.assertEqual(max_active, 2)

    async def test_chunk_initial_context_is_scoped_to_same_file(self):
        state = ExtractionRunState(
            chunk_results=[
                ExtractionChunkResult(
                    chunk_index=0,
                    file_path="README.md",
                    start_idx=0,
                    end_idx=0,
                    status="completed",
                    extraction_context=resource_context(
                        "readme-context",
                        "README context.",
                    ),
                ),
                ExtractionChunkResult(
                    chunk_index=1,
                    file_path="data.csv",
                    start_idx=0,
                    end_idx=0,
                    status="completed",
                    extraction_context=resource_context(
                        "data-context",
                        "Data file context.",
                    ),
                ),
            ],
        )

        context = ExtractionService._merged_completed_chunk_context_or_none(
            state,
            file_path="data.csv",
        )

        self.assertIsNotNone(context)
        self.assertEqual(
            [resource.identifier for resource in context.resources],
            ["data-context"],
        )

    async def test_chunk_initial_context_for_prompt_is_full_when_previous_usage_is_under_threshold(self):
        service, _, _ = make_service([[make_chunk()]])
        service.settings.initial_extraction_context_token_threshold = 100
        state = ExtractionRunState(
            chunk_results=[
                ExtractionChunkResult(
                    chunk_index=index,
                    file_path="README.md",
                    start_idx=index,
                    end_idx=index,
                    status="completed",
                    context_tokens=20,
                    extraction_context=resource_context(
                        f"resource-{index}",
                        f"Verbose metadata context {index} " * 10,
                    ),
                )
                for index in range(4)
            ],
        )

        context = service._initial_extraction_context_for_prompt(
            state,
            file_path="README.md",
            current_chunk_index=4,
        )

        self.assertIsNotNone(context)
        self.assertEqual(
            [resource.identifier for resource in context.resources],
            ["resource-0", "resource-1", "resource-2", "resource-3"],
        )

    async def test_chunk_initial_context_for_prompt_is_capped_after_high_previous_usage(self):
        service, _, _ = make_service([[make_chunk()]])
        service.settings.initial_extraction_context_token_threshold = 50
        state = ExtractionRunState(
            chunk_results=[
                ExtractionChunkResult(
                    chunk_index=index,
                    file_path="README.md",
                    start_idx=index,
                    end_idx=index,
                    status="completed",
                    context_tokens=100 if index == 3 else 20,
                    extraction_context=resource_context(
                        f"resource-{index}",
                        f"Verbose metadata context {index} " * 10,
                    ),
                )
                for index in range(4)
            ],
        )

        context = service._initial_extraction_context_for_prompt(
            state,
            file_path="README.md",
            current_chunk_index=4,
        )

        self.assertIsNotNone(context)
        self.assertLess(len(context.resources), 4)
        self.assertEqual(context.resources[-1].identifier, "resource-3")

    async def test_pause_extraction_cancels_task_and_marks_running_chunk_pending(self):
        service, task_registry, output_repository = make_service([[make_chunk()]])
        extraction_started = asyncio.Event()

        async def fake_generate(*_args, **kwargs):
            output_type = kwargs["output_type"]
            if output_type is ExtractionContext:
                extraction_started.set()
                await asyncio.Event().wait()
            return CompletionResult(
                output={"id": "dataset"},
                usage=RunUsage(requests=1, input_tokens=30, output_tokens=8),
            )

        with patch("app.services.extraction_service.generate_structured", side_effect=fake_generate):
            result, status = await service.run_extraction(
                data_package_id="package-id",
                profile_identifier="profile",
            )
            self.assertIsNone(result)
            self.assertEqual(status, TaskStatus.RUNNING)
            await asyncio.wait_for(extraction_started.wait(), timeout=2)

            status, progress = await service.pause_extraction(data_package_id="package-id")

        self.assertEqual(status, TaskStatus.CANCELLED)
        self.assertIsNotNone(progress)
        self.assertEqual(progress.stage, "paused")
        self.assertIsNotNone(output_repository.run_state)
        self.assertEqual(
            [chunk.status for chunk in output_repository.run_state.chunk_results],
            ["pending"],
        )
        self.assertEqual(
            task_registry.get_task_info("extraction:run:package-id").status,
            TaskStatus.CANCELLED,
        )


    async def test_object_grounding_normalization_picks_defined_term_from_concept_candidates(self):
        service, _, _ = make_service([[make_chunk()]])
        state = ExtractionRunState()
        discovery = _ObjectGroundingCandidateDiscovery(
            object_identifier="ir-spectrum",
            object_kind="Resource",
            raw_type="spectrum",
            source_context={"identifier": "ir-spectrum", "type": "spectrum"},
            query_ids=["voc4cat-concept", "voc4cat-collection"],
        )
        state.vocab_queries.append(
            ExtractionVocabQueryRecord(
                query_id="voc4cat-concept",
                kind="object_grounding",
                source_value="ir-spectrum",
                source_context={"identifier": "ir-spectrum", "type": "spectrum"},
                vocabulary_identifier="https://w3id.org/nfdi4cat/voc4cat",
                rdf_type="skos__Concept",
                query=VocabQuery(rdf_type="skos__Concept", fulltext_query="spectrum"),
                status="completed",
                result=VocabQueryResult(
                    identifier="https://w3id.org/nfdi4cat/voc4cat",
                    rdf_type="skos__Concept",
                    resources={
                        "https://w3id.org/nfdi4cat/voc4cat_42": CompactVocabResource(
                            uri="https://w3id.org/nfdi4cat/voc4cat_42",
                            rdf_types=["skos__Concept"],
                            properties={"skos__prefLabel": "IR spectrum"},
                        ),
                        "https://w3id.org/nfdi4cat/voc4cat_99": CompactVocabResource(
                            uri="https://w3id.org/nfdi4cat/voc4cat_99",
                            rdf_types=["skos__Concept"],
                            properties={"skos__prefLabel": "catalyst sample"},
                        ),
                    },
                ),
            )
        )
        state.vocab_queries.append(
            ExtractionVocabQueryRecord(
                query_id="voc4cat-collection",
                kind="object_grounding",
                source_value="ir-spectrum",
                source_context={"identifier": "ir-spectrum", "type": "spectrum"},
                vocabulary_identifier="https://w3id.org/nfdi4cat/voc4cat",
                rdf_type="skos__Collection",
                query=VocabQuery(rdf_type="skos__Collection", fulltext_query="spectrum"),
                status="completed",
                result=VocabQueryResult(
                    identifier="https://w3id.org/nfdi4cat/voc4cat",
                    rdf_type="skos__Collection",
                    resources={
                        "https://w3id.org/nfdi4cat/voc4cat_coll_1": CompactVocabResource(
                            uri="https://w3id.org/nfdi4cat/voc4cat_coll_1",
                            rdf_types=["skos__Collection"],
                            properties={"skos__prefLabel": "spectroscopy collection"},
                        ),
                    },
                ),
            )
        )

        async def fake_generate(*_args, **kwargs):
            return CompletionResult(
                output=VocabularyCandidateSelection(
                    selected_uri="https://w3id.org/nfdi4cat/voc4cat_42",
                    confidence=0.9,
                    reason="IR spectrum matches the prefLabel of the candidate concept.",
                ),
                usage=RunUsage(requests=1),
            )

        trace = TracedExtractionObject(
            object_kind="Resource",
            extracted_object=Resource(
                identifier="ir-spectrum",
                description="IR spectrum measurement",
                type="spectrum",
            ),
            source_text="IR spectrum measurement",
        )
        with patch("app.services.extraction_service.generate_structured", side_effect=fake_generate):
            grounded = await service._normalize_object_grounding_from_candidates(
                data_package_id="package-id",
                state=state,
                discovery=discovery,
                trace=trace,
                selection_semaphore=asyncio.Semaphore(1),
                warnings=[],
            )

        self.assertIsInstance(grounded, GroundedExtractionObject)
        self.assertEqual(grounded.object_identifier, "ir-spectrum")
        self.assertEqual(grounded.object_kind, "Resource")
        self.assertEqual(grounded.source_value, "spectrum")
        self.assertIsNotNone(grounded.defined_term)
        self.assertEqual(
            grounded.defined_term.id,
            "https://w3id.org/nfdi4cat/voc4cat_42",
        )
        self.assertEqual(grounded.defined_term.title, "IR spectrum")
        self.assertEqual(
            grounded.defined_term.from_CV,
            "https://w3id.org/nfdi4cat/voc4cat",
        )
        self.assertGreater(grounded.confidence, 0.0)

    async def test_object_grounding_normalization_keeps_raw_type_when_selector_returns_null(self):
        service, _, _ = make_service([[make_chunk()]])
        state = ExtractionRunState()
        discovery = _ObjectGroundingCandidateDiscovery(
            object_identifier="mystery-object",
            object_kind="Resource",
            raw_type="unknown",
            source_context={"identifier": "mystery-object", "type": "unknown"},
            query_ids=["voc4cat-concept"],
        )
        state.vocab_queries.append(
            ExtractionVocabQueryRecord(
                query_id="voc4cat-concept",
                kind="object_grounding",
                source_value="mystery-object",
                source_context={"identifier": "mystery-object", "type": "unknown"},
                vocabulary_identifier="https://w3id.org/nfdi4cat/voc4cat",
                rdf_type="skos__Concept",
                query=VocabQuery(rdf_type="skos__Concept", fulltext_query="unknown"),
                status="completed",
                result=VocabQueryResult(
                    identifier="https://w3id.org/nfdi4cat/voc4cat",
                    rdf_type="skos__Concept",
                    resources={
                        "https://w3id.org/nfdi4cat/voc4cat_77": CompactVocabResource(
                            uri="https://w3id.org/nfdi4cat/voc4cat_77",
                            rdf_types=["skos__Concept"],
                            properties={"skos__prefLabel": "photocatalysis"},
                        )
                    },
                ),
            )
        )

        async def fake_generate(*_args, **_kwargs):
            return CompletionResult(
                output=VocabularyCandidateSelection(
                    selected_uri=None,
                    confidence=0.0,
                    reason="No candidate matches the object description.",
                ),
                usage=RunUsage(requests=1),
            )

        trace = TracedExtractionObject(
            object_kind="Resource",
            extracted_object=Resource(
                identifier="mystery-object",
                description="mystery",
                type="unknown",
            ),
            source_text="mystery",
        )
        with patch("app.services.extraction_service.generate_structured", side_effect=fake_generate):
            grounded = await service._normalize_object_grounding_from_candidates(
                data_package_id="package-id",
                state=state,
                discovery=discovery,
                trace=trace,
                selection_semaphore=asyncio.Semaphore(1),
                warnings=[],
            )

        self.assertEqual(grounded.object_identifier, "mystery-object")
        self.assertIsNone(grounded.defined_term)
        self.assertEqual(grounded.source_value, "unknown")

    async def test_object_grounding_normalization_keeps_raw_type_when_concept_candidates_are_empty(self):
        service, _, _ = make_service([[make_chunk()]])
        state = ExtractionRunState()
        discovery = _ObjectGroundingCandidateDiscovery(
            object_identifier="spectrum",
            object_kind="Resource",
            raw_type="spectrum",
            source_context={"identifier": "spectrum", "type": "spectrum"},
            query_ids=["voc4cat-collection-only"],
        )
        state.vocab_queries.append(
            ExtractionVocabQueryRecord(
                query_id="voc4cat-collection-only",
                kind="object_grounding",
                source_value="spectrum",
                source_context={"identifier": "spectrum", "type": "spectrum"},
                vocabulary_identifier="https://w3id.org/nfdi4cat/voc4cat",
                rdf_type="skos__Collection",
                query=VocabQuery(rdf_type="skos__Collection", fulltext_query="spectrum"),
                status="completed",
                result=VocabQueryResult(
                    identifier="https://w3id.org/nfdi4cat/voc4cat",
                    rdf_type="skos__Collection",
                    resources={},
                ),
            )
        )

        trace = TracedExtractionObject(
            object_kind="Resource",
            extracted_object=Resource(
                identifier="spectrum",
                description="spectrum",
                type="spectrum",
            ),
            source_text="spectrum",
        )
        grounded = await service._normalize_object_grounding_from_candidates(
            data_package_id="package-id",
            state=state,
            discovery=discovery,
            trace=trace,
            selection_semaphore=asyncio.Semaphore(1),
            warnings=[],
        )

        self.assertEqual(grounded.source_value, "spectrum")
        self.assertIsNone(grounded.defined_term)

    async def test_normalize_from_state_vocab_queries_rebuilds_object_grounding(self):
        service, _, _ = make_service([[make_chunk()]])
        state = ExtractionRunState()
        state.vocab_queries.append(
            ExtractionVocabQueryRecord(
                query_id="voc4cat-concept",
                kind="object_grounding",
                source_value="dataset",
                source_context={"identifier": "dataset", "type": "dataset", "object_kind": "Resource"},
                vocabulary_identifier="https://w3id.org/nfdi4cat/voc4cat",
                rdf_type="skos__Concept",
                query=VocabQuery(rdf_type="skos__Concept", fulltext_query="dataset"),
                status="completed",
                result=VocabQueryResult(
                    identifier="https://w3id.org/nfdi4cat/voc4cat",
                    rdf_type="skos__Concept",
                    resources={
                        "https://w3id.org/nfdi4cat/voc4cat_5": CompactVocabResource(
                            uri="https://w3id.org/nfdi4cat/voc4cat_5",
                            rdf_types=["skos__Concept"],
                            properties={"skos__prefLabel": "dataset"},
                        )
                    },
                ),
            )
        )

        async def fake_generate(*_args, **_kwargs):
            return CompletionResult(
                output=VocabularyCandidateSelection(
                    selected_uri="https://w3id.org/nfdi4cat/voc4cat_5",
                    confidence=0.8,
                    reason="Direct match.",
                ),
                usage=RunUsage(requests=1),
            )

        extraction_context = ExtractionContext(
            extraction_objects=[
                TracedExtractionObject(
                    object_kind="Resource",
                    extracted_object=Resource(
                        identifier="dataset",
                        description="dataset",
                        type="dataset",
                    ),
                    source_text="dataset",
                )
            ]
        )
        with patch("app.services.extraction_service.generate_structured", side_effect=fake_generate):
            normalization = await service._normalize_from_state_vocab_queries(
                data_package_id="package-id",
                state=state,
                extraction_context=extraction_context,
                warnings=[],
            )

        self.assertEqual(len(normalization.grounded_objects), 1)
        grounding = normalization.grounded_objects[0]
        self.assertIsInstance(grounding, GroundedExtractionObject)
        self.assertEqual(grounding.object_identifier, "dataset")
        self.assertEqual(grounding.object_kind, "Resource")
        self.assertEqual(grounding.source_value, "dataset")
        self.assertIsNotNone(grounding.defined_term)
        self.assertEqual(grounding.defined_term.id, "https://w3id.org/nfdi4cat/voc4cat_5")

    async def test_normalize_from_candidate_tasks_wraps_trace_in_grounded_extraction_object(self):
        service, _, _ = make_service([[make_chunk()]])
        extraction_context = ExtractionContext(
            extraction_objects=[
                TracedExtractionObject(
                    object_kind="Resource",
                    extracted_object=Resource(
                        identifier="ir-spectrum",
                        description="IR spectrum measurement",
                        type="spectrum",
                    ),
                    source_text="IR spectrum measurement",
                )
            ]
        )
        state = ExtractionRunState()
        discovery = _ObjectGroundingCandidateDiscovery(
            object_identifier="ir-spectrum",
            object_kind="Resource",
            raw_type="spectrum",
            source_context={"identifier": "ir-spectrum", "type": "spectrum"},
            query_ids=["voc4cat-concept"],
        )
        state.vocab_queries.append(
            ExtractionVocabQueryRecord(
                query_id="voc4cat-concept",
                kind="object_grounding",
                source_value="ir-spectrum",
                source_context={"identifier": "ir-spectrum", "type": "spectrum"},
                vocabulary_identifier="https://w3id.org/nfdi4cat/voc4cat",
                rdf_type="skos__Concept",
                query=VocabQuery(rdf_type="skos__Concept", fulltext_query="spectrum"),
                status="completed",
                result=VocabQueryResult(
                    identifier="https://w3id.org/nfdi4cat/voc4cat",
                    rdf_type="skos__Concept",
                    resources={
                        "https://w3id.org/nfdi4cat/voc4cat_42": CompactVocabResource(
                            uri="https://w3id.org/nfdi4cat/voc4cat_42",
                            rdf_types=["skos__Concept"],
                            properties={"skos__prefLabel": "IR spectrum"},
                        )
                    },
                ),
            )
        )

        async def fake_generate(*_args, **_kwargs):
            return CompletionResult(
                output=VocabularyCandidateSelection(
                    selected_uri="https://w3id.org/nfdi4cat/voc4cat_42",
                    confidence=0.9,
                    reason="IR spectrum prefLabel matches.",
                ),
                usage=RunUsage(requests=1),
            )

        async def fake_discover(*_args, **_kwargs):
            return discovery

        task = asyncio.create_task(fake_discover())

        with patch("app.services.extraction_service.generate_structured", side_effect=fake_generate):
            normalization = await service._normalize_from_candidate_tasks(
                data_package_id="package-id",
                state=state,
                candidate_tasks=[task],
                extraction_context=extraction_context,
                warnings=[],
            )

        self.assertEqual(len(normalization.grounded_objects), 1)
        grounded = normalization.grounded_objects[0]
        self.assertIsInstance(grounded, GroundedExtractionObject)
        self.assertEqual(grounded.object_identifier, "ir-spectrum")
        self.assertEqual(grounded.object_kind, "Resource")
        self.assertEqual(grounded.source_value, "spectrum")
        self.assertIsInstance(grounded.extracted_object, Resource)
        self.assertEqual(grounded.extracted_object.identifier, "ir-spectrum")
        self.assertEqual(grounded.extracted_object.type, "spectrum")
        self.assertIsNotNone(grounded.defined_term)
        self.assertEqual(grounded.defined_term.id, "https://w3id.org/nfdi4cat/voc4cat_42")
        self.assertEqual(grounded.defined_term.title, "IR spectrum")
        self.assertEqual(grounded.defined_term.from_CV, "https://w3id.org/nfdi4cat/voc4cat")
        # The original ExtractionContext and its traces are unchanged.
        self.assertIsNone(extraction_context.extraction_objects[0].extracted_object.model_extra)
        # grounded_objects and object_groundings stay in sync.
        self.assertEqual(
            [g.object_identifier for g in normalization.object_groundings],
            ["ir-spectrum"],
        )

    async def test_grounded_extraction_object_keeps_raw_type_when_selector_returns_null(self):
        service, _, _ = make_service([[make_chunk()]])
        extraction_context = ExtractionContext(
            extraction_objects=[
                TracedExtractionObject(
                    object_kind="Resource",
                    extracted_object=Resource(
                        identifier="mystery",
                        description="mystery",
                        type="unknown",
                    ),
                    source_text="mystery",
                )
            ]
        )
        state = ExtractionRunState()
        discovery = _ObjectGroundingCandidateDiscovery(
            object_identifier="mystery",
            object_kind="Resource",
            raw_type="unknown",
            source_context={"identifier": "mystery", "type": "unknown"},
            query_ids=["voc4cat-concept"],
        )
        state.vocab_queries.append(
            ExtractionVocabQueryRecord(
                query_id="voc4cat-concept",
                kind="object_grounding",
                source_value="mystery",
                source_context={"identifier": "mystery", "type": "unknown"},
                vocabulary_identifier="https://w3id.org/nfdi4cat/voc4cat",
                rdf_type="skos__Concept",
                query=VocabQuery(rdf_type="skos__Concept", fulltext_query="unknown"),
                status="completed",
                result=VocabQueryResult(
                    identifier="https://w3id.org/nfdi4cat/voc4cat",
                    rdf_type="skos__Concept",
                    resources={
                        "https://w3id.org/nfdi4cat/voc4cat_77": CompactVocabResource(
                            uri="https://w3id.org/nfdi4cat/voc4cat_77",
                            rdf_types=["skos__Concept"],
                            properties={"skos__prefLabel": "photocatalysis"},
                        )
                    },
                ),
            )
        )

        async def fake_generate(*_args, **_kwargs):
            return CompletionResult(
                output=VocabularyCandidateSelection(
                    selected_uri=None,
                    confidence=0.0,
                    reason="No candidate matches.",
                ),
                usage=RunUsage(requests=1),
            )

        async def fake_discover(*_args, **_kwargs):
            return discovery

        task = asyncio.create_task(fake_discover())

        with patch("app.services.extraction_service.generate_structured", side_effect=fake_generate):
            normalization = await service._normalize_from_candidate_tasks(
                data_package_id="package-id",
                state=state,
                candidate_tasks=[task],
                extraction_context=extraction_context,
                warnings=[],
            )

        self.assertEqual(len(normalization.grounded_objects), 1)
        grounded = normalization.grounded_objects[0]
        self.assertIsNone(grounded.defined_term)
        self.assertEqual(grounded.source_value, "unknown")
        self.assertEqual(grounded.extracted_object.type, "unknown")


if __name__ == "__main__":
    unittest.main()
