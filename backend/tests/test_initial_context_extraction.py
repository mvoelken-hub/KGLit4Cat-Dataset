import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core.task_registry import TaskRegistry, TaskStatus, TaskType
from app.domain.datasources import ContentChunk, DataPackage, FileEntry
from app.domain.extraction import (
    ChunkingRequiredError,
    ExtractionChunkResult,
    ExtractionContext,
    ExtractionNormalization,
    ExtractionRunResult,
    ExtractionRunState,
    ExtractionVocabQueryConfig,
    ExtractionVocabQueryRecord,
    RankedFile,
    VocabularyCandidateSelection,
)
from app.domain.semantics import CompactVocabResource, VocabQuery, VocabQueryResult, VocabSchemeInfo, VocabTermScheme
from app.ollama.completion import CompletionResult
from app.ollama.errors import MaxRetriesExceeded, OutputParsingError
from app.ollama.usage import RunUsage
from app.services.extraction_service import ExtractionService, _QualitativeCandidateDiscovery


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
        return SimpleNamespace(identifier=identifier, target_class="Dataset")

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
        self.result: ExtractionRunResult | None = None
        self.run_state: ExtractionRunState | None = None
        self.warnings: list[str] = []
        self.token_usage: dict[str, dict[str, int]] = {}

    def save_extraction_context(self, *, workflow_id: str, extraction_context: ExtractionContext):
        self.context = extraction_context
        self.contexts.append(extraction_context)

    def load_extraction_context(self, workflow_id: str) -> ExtractionContext:
        if self.context is None:
            raise FileNotFoundError
        return self.context

    def save_extraction_result(self, *, workflow_id: str, result: ExtractionRunResult):
        self.result = result

    def load_extraction_result(self, workflow_id: str) -> ExtractionRunResult:
        if self.result is None:
            raise FileNotFoundError
        return self.result

    def save_extraction_run_state(self, *, workflow_id: str, state: ExtractionRunState):
        self.run_state = state

    def load_extraction_run_state(self, workflow_id: str) -> ExtractionRunState:
        if self.run_state is None:
            raise FileNotFoundError
        return self.run_state

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
        self.result = None
        self.run_state = None
        self.warnings = []
        self.token_usage = {}


def make_chunk(start_idx: int = 0, content: str = "sample measured at 20 C") -> ContentChunk:
    return ContentChunk(
        content=content,
        data_package_id="package-id",
        file_path="README.md",
        start_idx=start_idx,
        end_idx=start_idx,
    )


def resource_context(identifier: str, description: str) -> ExtractionContext:
    return ExtractionContext.model_validate(
        {
            "extraction_objects": [
                {
                    "object_type": "resource",
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
                    "object_type": "resource",
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


def make_service(chunks_by_file: list[list[ContentChunk]]):
    task_registry = TaskRegistry(SimpleNamespace(), FakeLogger())  # type: ignore[arg-type]
    output_repository = FakeOutputRepository()
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
            document={"id": "old"},
            extraction_context=ExtractionContext(),
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

    async def test_profile_projection_falls_back_to_minimal_valid_document(self):
        service, _, output_repository = make_service([[make_chunk()]])
        service.profile_service = TitleProfileService()  # type: ignore[assignment]
        context = ExtractionContext.model_validate(
            {
                "extraction_objects": [
                    {
                        "object_type": "evaluated_entity",
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

        with patch(
            "app.services.extraction_service.generate_structured",
            side_effect=MaxRetriesExceeded(),
        ):
            result = await service._project_and_save_result(
                data_package_id="package-id",
                profile_identifier="profile",
                profile_target_class="Dataset",
                extraction_context=context,
                normalization=ExtractionNormalization(),
                validation_schema=TitleProfileService.schema,
                warnings=warnings,
            )

        self.assertEqual(result.document, {"title": "SG-V4050", "identifier": "package-id"})
        self.assertIn("Profile projection fell back", result.warnings[0])
        self.assertEqual(output_repository.result, result)

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
        self.assertEqual(output_repository.result.document["id"], "dataset")
        self.assertIn("chunk_extraction", output_repository.token_usage)

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
                output=resource_context("resource-one", "First partial resource."),
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
            if output_type is ExtractionContext:
                call_order.append(f"extract:{len(call_order)}")
                output = chunk_outputs.pop(0)
                if isinstance(output, Exception):
                    raise output
                return output
            call_order.append("profile")
            return CompletionResult(
                output={"id": "dataset"},
                usage=RunUsage(requests=1, input_tokens=30, output_tokens=8),
            )

        async def fake_repair(*_args, **_kwargs):
            call_order.append("repair")
            return CompletionResult(
                output=resource_context("resource-two", "Repaired resource."),
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

        self.assertEqual(call_order, ["extract:0", "extract:1", "repair", "profile"])
        self.assertEqual(
            [resource.identifier for resource in output_repository.context.resources],
            ["resource-one", "resource-two", "README.md"],
        )
        self.assertIn("chunk_extraction_repair", output_repository.token_usage)

    async def test_resume_reuses_completed_chunk_results(self):
        service, task_registry, output_repository = make_service(
            [[make_chunk(0, "sample one"), make_chunk(1, "sample two")]]
        )
        output_repository.save_extraction_run_state(
            workflow_id="package-id",
            state=ExtractionRunState(
                ranked_files=[RankedFile(rank=1, file_path="README.md")],
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

        self.assertEqual(outputs, [])
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

    async def test_vocab_candidate_discovery_starts_after_completed_chunk_before_all_chunks_finish(self):
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
                    await asyncio.wait_for(semantic_service.discovery_started.wait(), timeout=1)
                    call_order.append("discovery-before-second-finished")
                call_order.append("extract")
                return CompletionResult(
                    output=qualitative_context("sample", "phase", "liquid"),
                    usage=RunUsage(requests=1),
                )
            call_order.append("profile")
            return CompletionResult(output={"id": "dataset"}, usage=RunUsage(requests=1))

        with patch("app.services.extraction_service.generate_structured", side_effect=fake_generate):
            await service.run_extraction(
                data_package_id="package-id",
                profile_identifier="profile",
            )
            await task_registry.wait_for_task("extraction:run:package-id", timeout=2)

        self.assertIn("discovery-before-second-finished", call_order)
        self.assertGreaterEqual(len(semantic_service.query_calls), 1)
        self.assertIsNotNone(output_repository.run_state)
        vocab_queries = output_repository.run_state.chunk_results[0].vocab_queries
        self.assertGreaterEqual(len(vocab_queries), 1)
        self.assertEqual(vocab_queries[0].status, "completed")
        self.assertIsNotNone(vocab_queries[0].query)
        self.assertIsNotNone(vocab_queries[0].result)

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
            state.chunk_results.append(
                ExtractionChunkResult(
                    chunk_index=index,
                    file_path="README.md",
                    start_idx=index,
                    end_idx=index,
                    status="completed",
                    vocab_queries=[
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
                    ],
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
        with patch("app.services.extraction_service.generate_structured", side_effect=fake_generate):
            normalization = await service._normalize_from_candidate_tasks(
                data_package_id="package-id",
                state=state,
                candidate_tasks=tasks,
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
            state.chunk_results.append(
                ExtractionChunkResult(
                    chunk_index=index,
                    file_path="README.md",
                    start_idx=index,
                    end_idx=index,
                    status="completed",
                    vocab_queries=[
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
                    ],
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
        with patch("app.services.extraction_service.generate_structured", side_effect=fake_generate):
            await service._normalize_from_candidate_tasks(
                data_package_id="package-id",
                state=state,
                candidate_tasks=tasks,
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


if __name__ == "__main__":
    unittest.main()
