# Graph Report - .  (2026-06-13)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 2257 nodes · 6602 edges · 99 communities (77 shown, 22 thin omitted)
- Extraction: 69% EXTRACTED · 31% INFERRED · 0% AMBIGUOUS · INFERRED: 2029 edges (avg confidence: 0.54)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `aefcf5ee`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 79|Community 79]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 81|Community 81]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 83|Community 83]]
- [[_COMMUNITY_Community 84|Community 84]]
- [[_COMMUNITY_Community 85|Community 85]]
- [[_COMMUNITY_Community 86|Community 86]]

## God Nodes (most connected - your core abstractions)
1. `ExtractionService` - 200 edges
2. `TaskStatus` - 180 edges
3. `OllamaClientWrapper` - 143 edges
4. `Settings` - 140 edges
5. `TaskRegistry` - 131 edges
6. `BaseModel` - 112 edges
7. `TaskType` - 109 edges
8. `DataSourceService` - 108 edges
9. `RunUsage` - 100 edges
10. `SemanticService` - 97 edges

## Surprising Connections (you probably didn't know these)
- `Exception` --uses--> `DataSourceService`  [INFERRED]
  backend/app/api/v1/datasources.py → backend/app/services/datasource_service.py
- `UploadFile` --uses--> `DataSourceService`  [INFERRED]
  backend/app/api/v1/datasources.py → backend/app/services/datasource_service.py
- `ChunkingRequest` --uses--> `DataSourceService`  [INFERRED]
  backend/app/api/v1/datasources.py → backend/app/services/datasource_service.py
- `Query` --uses--> `DataSourceService`  [INFERRED]
  backend/app/api/v1/datasources.py → backend/app/services/datasource_service.py
- `description` --uses--> `DataSourceService`  [INFERRED]
  backend/app/api/v1/datasources.py → backend/app/services/datasource_service.py

## Import Cycles
- 1-file cycle: `backend/app/main.py -> backend/app/main.py`
- 1-file cycle: `backend/app/domain/datasources/datasource.py -> backend/app/domain/datasources/datasource.py`

## Communities (99 total, 22 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (105): import_initial_vocab(), InitialTasks, pull_ollama_models(), start_setup(), Any, DataSourceService, UploadFile, Exception (+97 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (7): Any, ExtractionContext, ExtractionRunProgress, ExtractionRunResult, TaskStatus, CompleteWorkflowProgress, ExtractionService

### Community 2 - "Community 2"
Cohesion: 0.05
Nodes (108): _api_get_ollama_config(), _api_is_reachable(), _api_patch_runtime(), _api_run_performance_test(), _backup_neo4j(), _bootstrap_vocabs(), _build_compose_cmd(), _check_command() (+100 more)

### Community 3 - "Community 3"
Cohesion: 0.03
Nodes (53): App(), asRecord(), asRecordArray(), buildTraceSegments(), BusyKey, ChunkingStatusPanel(), ChunkTraceModal(), coerceVocabQueryResult() (+45 more)

### Community 4 - "Community 4"
Cohesion: 0.05
Nodes (26): FullTextIndexInfo, Graph, Neo4jDriver, OllamaClientWrapper, TraversalDirection, VectorIndexInfo, VocabGraphStatement, VocabResource (+18 more)

### Community 5 - "Community 5"
Cohesion: 0.06
Nodes (43): FullTextIndexInfo, Graph, Logger, Settings, VectorIndexInfo, Any, FullTextIndexInfo, VectorIndexInfo (+35 more)

### Community 6 - "Community 6"
Cohesion: 0.10
Nodes (68): AttributeMatch, Path, Any, EvaluationRunManifest, Path, Any, EvaluationRunManifest, ExtractionContext (+60 more)

### Community 7 - "Community 7"
Cohesion: 0.06
Nodes (29): BytesIO, BytesIO, ContentChunk, DataPackage, Path, BytesIO, DataPackage, FileEntry (+21 more)

### Community 8 - "Community 8"
Cohesion: 0.09
Nodes (20): Any, CurationLedgerRecord, DraftValidationResult, ExtractionContext, ExtractionFileSummary, ExtractionOverview, ExtractionOverviewStatus, ExtractionRunResult (+12 more)

### Community 9 - "Community 9"
Cohesion: 0.04
Nodes (50): arrayOfRecords(), CurationLedgerRecord, DraftQualityState, DraftValidationIssue, DraftValidationResult, emptyInitialContext(), emptyReviewState(), extractInitialContext() (+42 more)

### Community 10 - "Community 10"
Cohesion: 0.10
Nodes (52): TaskInfo, Any, Neo4jDriver, OllamaClientWrapper, Settings, Any, Exception, OllamaClientWrapper (+44 more)

### Community 11 - "Community 11"
Cohesion: 0.08
Nodes (33): Graph, HttpUrl, VocabResource, LoadedRdfGraph, Node, A vocabulary resource projection used for embedding generation., Domain-level representation of a vocabulary scheme., VocabAlreadyExistsError (+25 more)

### Community 12 - "Community 12"
Cohesion: 0.12
Nodes (11): TaskInfo, VocabResource, VocabSchemeInfo, FakeOllamaClient, FakeSemanticGraphRepository, FakeSettings, FakeTaskRegistry, make_service() (+3 more)

### Community 13 - "Community 13"
Cohesion: 0.07
Nodes (36): ApiError, formatApiDetail(), parseJsonResponse(), readJson(), applyCurationFieldAction(), getInitialContextProgress(), getPatchProgress(), getTokenUsage() (+28 more)

### Community 14 - "Community 14"
Cohesion: 0.13
Nodes (27): Any, QualitativeAttribute, QuantitativeAttribute, VocabQuery, BaseExtractionModel, DefinedTerm, QualitativeAttribute, QuantitativeAttribute (+19 more)

### Community 15 - "Community 15"
Cohesion: 0.17
Nodes (11): ExtractionNormalization, ExtractionRunState, VocabQuery, Semaphore, _ObjectGroundingCandidateDiscovery, _ProfileFieldCandidateDiscovery, _QualitativeCandidateDiscovery, _QuantityCandidateDiscovery (+3 more)

### Community 16 - "Community 16"
Cohesion: 0.08
Nodes (9): Any, CurationLedgerRecord, DraftValidationResult, ExtractionContext, ExtractionRunResult, ExtractionRunState, FieldCompletionLedgerRecord, ProjectionLedgerRecord (+1 more)

### Community 17 - "Community 17"
Cohesion: 0.13
Nodes (4): ExtractionFileSummary, ExtractionOverview, ExtractionChunkResult, FileRankingResult

### Community 18 - "Community 18"
Cohesion: 0.11
Nodes (24): defaultValueForSchema(), formatPrimitivePreview(), formatTreeNodeLabel(), getValueAtPath(), isPathProtected(), isRecord(), isTopLevelPath(), itemSchema() (+16 more)

### Community 19 - "Community 19"
Cohesion: 0.19
Nodes (9): IsolatedAsyncioTestCase, generate_structured(), Single /api/generate call with format=json_schema, parse + validate + retry., FakeGenerateResponse, FakeOllamaClient, GenerateStructuredHappyPathTests, GenerateStructuredRetryTests, Fake ollama GenerateResponse for testing. (+1 more)

### Community 20 - "Community 20"
Cohesion: 0.10
Nodes (6): FakeAsyncOllamaApi, FakeCpuEmbeddingSettings, FakeLogger, FakeSettings, OllamaClientWrapperAsyncTests, OllamaClientWrapperTests

### Community 21 - "Community 21"
Cohesion: 0.09
Nodes (26): buildQuery(), chunkDataPackage(), deleteDataPackage(), getChunkStatus(), getDataPackageChunks(), getFileEntryContent(), listDataPackages(), uploadDataPackage() (+18 more)

### Community 22 - "Community 22"
Cohesion: 0.12
Nodes (29): Exception, Form, HttpUrl, ProfileService, UploadFile, JsonLdExportResult, ProfileManifest, ProfileValidationResult (+21 more)

### Community 23 - "Community 23"
Cohesion: 0.29
Nodes (28): ExtractionFileSummary, ExtractionOverview, ExtractionOverviewStatus, ExtractionOverviewStatus, Exception, ExtractionContext, Context for metadata extraction, including activities, entities, datasets, and q, RankedFile (+20 more)

### Community 24 - "Community 24"
Cohesion: 0.10
Nodes (23): checkVocabularyEmbeddings(), deleteVocabulary(), getVocabulary(), importVocabulary(), listVocabularies(), queryVocabulary(), ContextActivity, ContextAgent (+15 more)

### Community 25 - "Community 25"
Cohesion: 0.11
Nodes (27): Any, Exception, OllamaClientWrapper, RunUsage, JsonSchema, MaxRetriesExceeded, ModelT, _estimated_tokens() (+19 more)

### Community 26 - "Community 26"
Cohesion: 0.10
Nodes (7): build_system_prompt_with_context(), DataGeneratingActivity, merge_extraction_context_results(), An experimental, measurement, acquisition, or processing activity that produces, A dataset resource or generated output, such as a file, dataset, spectrum, peak, Resource, ExtractionDomainTests

### Community 27 - "Community 27"
Cohesion: 0.17
Nodes (27): Exception, ExtractionService, ExtractionRunProgress, TaskStatus, ExtractionProgressResponse, ExtractionRunResult, _extraction_result_response(), _extraction_run_response() (+19 more)

### Community 28 - "Community 28"
Cohesion: 0.19
Nodes (5): DataPackage, CompactVocabResource, ExtractionServiceWorkflowTests, make_chunk(), make_service()

### Community 29 - "Community 29"
Cohesion: 0.14
Nodes (10): ContentChunk, DataPackage, DataSourceService, TaskStatus, DataSourceServiceTests, FakeBlobRepository, FakeSettings, FakeTaskRegistry (+2 more)

### Community 30 - "Community 30"
Cohesion: 0.12
Nodes (22): VocabQuery, VocabQueryResult, Form, HttpUrl, SemanticService, UploadFile, CompactVocabResourceResponse, _vocab_query_result_response() (+14 more)

### Community 31 - "Community 31"
Cohesion: 0.15
Nodes (23): Any, detect_enrichable_fields(), export_document_to_jsonld(), _format_json_path(), generate_profile_artifacts(), GeneratedProfileArtifacts, InvalidProfileIdentifierError, JsonLdExportResult (+15 more)

### Community 33 - "Community 33"
Cohesion: 0.18
Nodes (22): Any, are_probably_same_item(), _attribute_keys(), cap_extraction_context_for_prompt(), _compact_object_bullet(), deduplicate_extraction_context_items(), deduplicate_list(), _fit_traced_extraction_object() (+14 more)

### Community 34 - "Community 34"
Cohesion: 0.12
Nodes (16): classify_text_line(), _entropy(), _features(), _has_many_control_chars(), _looks_like_structured_text(), Text quality gate for filtering semantically meaningful lines.  This module is, Contains patterns like:     - key: value     - key = value     - Markdown hea, TextQualityDecision (+8 more)

### Community 35 - "Community 35"
Cohesion: 0.14
Nodes (12): Any, RunUsage, CompletionError, EmptyResponseError, MaxRetriesExceeded, OutputParsingError, Custom structured completion errors., JSON decode or schema validation failure. (+4 more)

### Community 36 - "Community 36"
Cohesion: 0.12
Nodes (12): Any, Exception, CompletionResult, _extract_json_schema(), Result of a successful structured completion., Extract a JSON Schema dict from a Pydantic model class or return a dict as-is., ExtractJsonSchemaTests, NestedOutput (+4 more)

### Community 37 - "Community 37"
Cohesion: 0.19
Nodes (5): TestClient, FakeRemoteSettings, FakeRunningModel, FakeSettings, TestSystemApi

### Community 38 - "Community 38"
Cohesion: 0.11
Nodes (7): Graph, TraversalDirection, VocabGraphStatement, VocabResource, VocabSchemeInfo, VocabSearchCandidate, SemanticGraphRepository

### Community 39 - "Community 39"
Cohesion: 0.20
Nodes (7): Any, GeneratedProfileArtifacts, Path, ProfileManifest, FileSystemProfileRepository, ProfileNotFoundError, FileSystemProfileRepositoryTests

### Community 40 - "Community 40"
Cohesion: 0.12
Nodes (14): main(), run_initial_vocab_bootstrap(), get_datasource_service(), get_extraction_service(), get_neo4j_driver(), get_ollama_client(), get_profile_service(), get_semantic_service() (+6 more)

### Community 41 - "Community 41"
Cohesion: 0.11
Nodes (18): compilerOptions, allowJs, allowSyntheticDefaultImports, esModuleInterop, forceConsistentCasingInFileNames, isolatedModules, jsx, lib (+10 more)

### Community 42 - "Community 42"
Cohesion: 0.22
Nodes (17): DataSourceService, Exception, UploadFile, DataPackage, ChunkingRequest, description, Query, _data_package_response() (+9 more)

### Community 43 - "Community 43"
Cohesion: 0.25
Nodes (12): FileEntry, attach_cosine_distances(), BufferWindow, combine_lines(), ContentChunk, _estimated_tokens(), FilteredLine, Combine each line with its neighbours using a sliding buffer window.      Retu (+4 more)

### Community 44 - "Community 44"
Cohesion: 0.15
Nodes (9): FileEntry, FakeFileEntry, ProtectedLineIndicesTests, Unit tests for the chunking domain logic, especially protected_line_indices., Line indices outside the file bounds are silently ignored., Minimal stand-in for FileEntry that only provides extracted content., When protected_line_indices contains [0], a dropped header is retained., With protected_line_indices=None the header is treated as usual. (+1 more)

### Community 45 - "Community 45"
Cohesion: 0.11
Nodes (17): dependencies, react, react-dom, devDependencies, @types/react, @types/react-dom, typescript, vite (+9 more)

### Community 46 - "Community 46"
Cohesion: 0.18
Nodes (14): BaseModel, InitialVocab, build_extraction_context_prompt(), ChunkContext, ChunkMetadata, ExtractionVocabQueryConfig, CompleteWorkflowProgressResponse, CompleteWorkflowRunResponse (+6 more)

### Community 47 - "Community 47"
Cohesion: 0.15
Nodes (7): CompletionResult, ProfilePatchDocument, empty_profile_patch(), FakeLogger, FakeProfileService, quantity_profile_patch(), type_profile_patch()

### Community 49 - "Community 49"
Cohesion: 0.16
Nodes (6): Any, GeneratedProfileArtifacts, ProfileManifest, ProfileRepository, Protocol, ProfileRepository

### Community 50 - "Community 50"
Cohesion: 0.20
Nodes (5): Any, Task, Raised when trying to create a task with a name that is already running., Creates and registers a new background task. If a task with the same name is alr, TaskStillRunningError

### Community 51 - "Community 51"
Cohesion: 0.24
Nodes (11): Any, ExtractionContext, ExtractionNormalization, TracedExtractionObject, A traced extraction object pairs an extracted class instance with the specific t, TracedExtractionObject, build_profile_patch_prompt(), build_profile_projection_prompt() (+3 more)

### Community 52 - "Community 52"
Cohesion: 0.27
Nodes (4): TestClient, FakeSemanticService, make_test_client(), SemanticApiRoutingTests

### Community 53 - "Community 53"
Cohesion: 0.22
Nodes (12): build_system_prompt_with_overview(), _same_file_memory_budget(), build_extraction_file_summary_prompt(), build_extraction_overview_fallback_prompt(), build_extraction_overview_prompt(), ExtractionFileContentWindow, ExtractionOverviewFilePreview, ExtractionOverviewFileRole (+4 more)

### Community 54 - "Community 54"
Cohesion: 0.36
Nodes (12): Path, doi_from_name(), field(), infer_key(), latex_clean(), main(), normalize(), normalize_arxiv() (+4 more)

### Community 56 - "Community 56"
Cohesion: 0.20
Nodes (4): ExtractionContext, ExtractionRunResult, qualitative_context(), quantitative_context()

### Community 57 - "Community 57"
Cohesion: 0.17
Nodes (6): AgenticEntity, EvaluatedEntity, Method, A method, plan, protocol, pulse sequence, acquisition procedure, processing rout, The actual target entity evaluated by a data-generating activity, such as a samp, An entity with agency that can perform activities, such as a person, organizatio

### Community 58 - "Community 58"
Cohesion: 0.27
Nodes (9): Tunable thresholds for the text-quality classifier.      All defaults mirror t, TextQualityConfig, ChunkingRequest, ChunkRequestResponse, ChunkResponse, DataPackageResponse, FileEntryContentResponse, FileEntryResponse (+1 more)

### Community 60 - "Community 60"
Cohesion: 0.24
Nodes (3): JsonLdExportResult, ProfileManifest, ProfileValidationResult

### Community 61 - "Community 61"
Cohesion: 0.47
Nodes (6): build_file_ranking_prompt(), fallback_file_ranking(), _fallback_score(), _file_extension(), FileContext, FileRankingResult

### Community 64 - "Community 64"
Cohesion: 0.25
Nodes (7): compilerOptions, allowSyntheticDefaultImports, composite, module, moduleResolution, skipLibCheck, include

### Community 66 - "Community 66"
Cohesion: 0.32
Nodes (8): decodeTraceEscapes(), findDirectRange(), findFlexibleWhitespaceRange(), findLineSequenceRange(), findTraceRange(), normalizeTraceSearchText(), stripTracePromptMetadata(), uniqueStrings()

### Community 68 - "Community 68"
Cohesion: 0.43
Nodes (3): Remove markdown code fences and surrounding whitespace., _strip_markdown_fences(), StripMarkdownFencesTests

### Community 71 - "Community 71"
Cohesion: 0.33
Nodes (4): AsyncClient, Logger, Settings, The underlying ollama.AsyncClient for direct /api/generate calls.          Use

### Community 72 - "Community 72"
Cohesion: 0.33
Nodes (4): GenerateResponse, _nanoseconds_to_milliseconds(), Token usage tracking for structured completions., Create RunUsage from an ollama GenerateResponse object.

### Community 78 - "Community 78"
Cohesion: 0.83
Nodes (3): compact_result(), load_api_key(), main()

## Knowledge Gaps
- **122 isolated node(s):** `ProfileManifest`, `ProfileValidationResult`, `ProfileValidationIssue`, `JsonLdExportResult`, `Path` (+117 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **22 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BaseModel` connect `Community 46` to `Community 0`, `Community 4`, `Community 5`, `Community 6`, `Community 7`, `Community 10`, `Community 11`, `Community 14`, `Community 19`, `Community 22`, `Community 23`, `Community 25`, `Community 27`, `Community 28`, `Community 30`, `Community 31`, `Community 33`, `Community 36`, `Community 43`, `Community 51`, `Community 53`, `Community 58`, `Community 61`?**
  _High betweenness centrality (0.181) - this node is a cross-community bridge._
- **Why does `ExtractionService` connect `Community 1` to `Community 0`, `Community 32`, `Community 67`, `Community 69`, `Community 6`, `Community 47`, `Community 48`, `Community 15`, `Community 16`, `Community 17`, `Community 56`, `Community 26`, `Community 27`, `Community 28`?**
  _High betweenness centrality (0.144) - this node is a cross-community bridge._
- **Why does `OllamaClientWrapper` connect `Community 0` to `Community 1`, `Community 2`, `Community 36`, `Community 71`, `Community 10`, `Community 46`, `Community 15`, `Community 17`, `Community 20`, `Community 86`, `Community 55`, `Community 25`, `Community 62`?**
  _High betweenness centrality (0.129) - this node is a cross-community bridge._
- **Are the 49 inferred relationships involving `ExtractionService` (e.g. with `Any` and `DataSourceService`) actually correct?**
  _`ExtractionService` has 49 INFERRED edges - model-reasoned connections that need verification._
- **Are the 176 inferred relationships involving `TaskStatus` (e.g. with `InitialTasks` and `Any`) actually correct?**
  _`TaskStatus` has 176 INFERRED edges - model-reasoned connections that need verification._
- **Are the 123 inferred relationships involving `OllamaClientWrapper` (e.g. with `InitialTasks` and `Any`) actually correct?**
  _`OllamaClientWrapper` has 123 INFERRED edges - model-reasoned connections that need verification._
- **Are the 129 inferred relationships involving `Settings` (e.g. with `InitialTasks` and `AsyncClient`) actually correct?**
  _`Settings` has 129 INFERRED edges - model-reasoned connections that need verification._