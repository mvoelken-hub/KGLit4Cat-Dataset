# Graph Report - .  (2026-06-15)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 2604 nodes · 8831 edges · 111 communities (84 shown, 27 thin omitted)
- Extraction: 64% EXTRACTED · 36% INFERRED · 0% AMBIGUOUS · INFERRED: 3186 edges (avg confidence: 0.54)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `25fd95ab`
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
- [[_COMMUNITY_Community 74|Community 74]]
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
- [[_COMMUNITY_Community 87|Community 87]]
- [[_COMMUNITY_Community 88|Community 88]]
- [[_COMMUNITY_Community 89|Community 89]]
- [[_COMMUNITY_Community 90|Community 90]]
- [[_COMMUNITY_Community 91|Community 91]]
- [[_COMMUNITY_Community 92|Community 92]]
- [[_COMMUNITY_Community 97|Community 97]]
- [[_COMMUNITY_Community 99|Community 99]]
- [[_COMMUNITY_Community 100|Community 100]]
- [[_COMMUNITY_Community 101|Community 101]]
- [[_COMMUNITY_Community 103|Community 103]]

## God Nodes (most connected - your core abstractions)
1. `ExtractionService` - 322 edges
2. `TaskStatus` - 207 edges
3. `Settings` - 168 edges
4. `OllamaClientWrapper` - 166 edges
5. `TaskRegistry` - 157 edges
6. `BaseModel` - 140 edges
7. `TaskType` - 135 edges
8. `RunUsage` - 135 edges
9. `DataSourceService` - 134 edges
10. `MaxRetriesExceeded` - 124 edges

## Surprising Connections (you probably didn't know these)
- `UploadFile` --uses--> `DataSourceService`  [INFERRED]
  backend/app/api/v1/datasources.py → backend/app/services/datasource_service.py
- `ChunkingRequest` --uses--> `DataSourceService`  [INFERRED]
  backend/app/api/v1/datasources.py → backend/app/services/datasource_service.py
- `Query` --uses--> `DataSourceService`  [INFERRED]
  backend/app/api/v1/datasources.py → backend/app/services/datasource_service.py
- `description` --uses--> `DataSourceService`  [INFERRED]
  backend/app/api/v1/datasources.py → backend/app/services/datasource_service.py
- `Exception` --uses--> `ProfileService`  [INFERRED]
  backend/app/api/v1/profiles.py → backend/app/services/profile_service.py

## Import Cycles
- 1-file cycle: `backend/app/main.py -> backend/app/main.py`
- 1-file cycle: `backend/app/domain/datasources/datasource.py -> backend/app/domain/datasources/datasource.py`

## Communities (111 total, 27 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (108): _api_get_ollama_config(), _api_is_reachable(), _api_patch_runtime(), _api_run_performance_test(), _backup_neo4j(), _bootstrap_vocabs(), _build_compose_cmd(), _check_command() (+100 more)

### Community 1 - "Community 1"
Cohesion: 0.03
Nodes (63): App(), asRecord(), asRecordArray(), buildTraceSegments(), BusyKey, ChunkingStatusPanel(), ChunkTraceModal(), coerceVocabQueryResult() (+55 more)

### Community 2 - "Community 2"
Cohesion: 0.05
Nodes (40): RunUsage, IsolatedAsyncioTestCase, _cut_repeated_line_run(), _estimated_tokens(), _extract_json_schema(), _failed_response_for_repair(), _fits_context_budget(), _format_json_path() (+32 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (31): Any, BaseException, CurationLedgerRecord, DraftValidationResult, EvidenceContext, ExtractionFileSummary, ExtractionOverview, ExtractionOverviewStatus (+23 more)

### Community 4 - "Community 4"
Cohesion: 0.11
Nodes (82): InitialTasks, pull_ollama_models(), start_setup(), Exception, TaskRegistry, Logger, Neo4jDriver, OllamaClientWrapper (+74 more)

### Community 5 - "Community 5"
Cohesion: 0.08
Nodes (64): Any, Exception, GenerateResponse, MaxRetriesExceeded, OllamaClientWrapper, PromptCompletionDiagnostics, RunUsage, ContentChunk (+56 more)

### Community 6 - "Community 6"
Cohesion: 0.10
Nodes (72): AttributeMatch, Path, Any, EvaluationRunManifest, Path, Any, EvaluationRunManifest, EvidenceContext (+64 more)

### Community 7 - "Community 7"
Cohesion: 0.05
Nodes (26): FullTextIndexInfo, Graph, Neo4jDriver, OllamaClientWrapper, TraversalDirection, VectorIndexInfo, VocabGraphStatement, VocabResource (+18 more)

### Community 8 - "Community 8"
Cohesion: 0.04
Nodes (56): arrayOfRecords(), ChunkRepairMode, CurationLedgerRecord, DraftQualityState, DraftValidationIssue, DraftValidationResult, emptyInitialContext(), emptyReviewState() (+48 more)

### Community 10 - "Community 10"
Cohesion: 0.18
Nodes (5): Any, Task, Raised when trying to create a task with a name that is already running., Creates and registers a new background task. If a task with the same name is alr, TaskStillRunningError

### Community 11 - "Community 11"
Cohesion: 0.12
Nodes (10): TaskInfo, VocabResource, FakeOllamaClient, FakeSemanticGraphRepository, FakeSettings, FakeTaskRegistry, make_service(), make_vocab() (+2 more)

### Community 12 - "Community 12"
Cohesion: 0.05
Nodes (5): evidence_text_match_score(), validate_evidence_context_for_chunk(), A dataset resource or generated output, such as a file, dataset, spectrum, peak, Resource, ExtractionDomainTests

### Community 13 - "Community 13"
Cohesion: 0.09
Nodes (33): Graph, HttpUrl, VocabResource, LoadedRdfGraph, Node, A vocabulary resource projection used for embedding generation., Domain-level representation of a vocabulary scheme., VocabAlreadyExistsError (+25 more)

### Community 14 - "Community 14"
Cohesion: 0.15
Nodes (3): ExtractionRunProgress, TaskStatus, ExtractionRunProgress

### Community 15 - "Community 15"
Cohesion: 0.06
Nodes (40): ApiError, formatApiDetail(), parseJsonResponse(), readJson(), getChunkStatus(), getDataPackageChunks(), listDataPackages(), uploadDataPackage() (+32 more)

### Community 17 - "Community 17"
Cohesion: 0.12
Nodes (30): Any, QualitativeAttribute, QuantitativeAttribute, VocabQuery, BaseExtractionModel, DefinedTerm, QualitativeAttribute, QuantitativeAttribute (+22 more)

### Community 18 - "Community 18"
Cohesion: 0.31
Nodes (36): Exception, Exception, Any, ExtractionOverviewStatus, EvidenceSignalLevel, Exception, EvidenceContext, FilteredEvidenceNote (+28 more)

### Community 19 - "Community 19"
Cohesion: 0.08
Nodes (32): buildQuery(), chunkDataPackage(), deleteDataPackage(), getFileEntryContent(), ChunkRequestResponse, ChunkResponse, ContextActivity, ContextAgent (+24 more)

### Community 21 - "Community 21"
Cohesion: 0.14
Nodes (9): BytesIO, ContentChunk, DataPackage, Path, BytesIO, FileSystemDataSourceBlobRepository, DataPackageTests, pdf_bytes() (+1 more)

### Community 22 - "Community 22"
Cohesion: 0.10
Nodes (6): FakeAsyncOllamaApi, FakeCpuEmbeddingSettings, FakeLogger, FakeSettings, OllamaClientWrapperAsyncTests, OllamaClientWrapperTests

### Community 23 - "Community 23"
Cohesion: 0.14
Nodes (31): Any, DataSourceService, ExtractionService, UploadFile, CompleteWorkflowProgressResponse, CompleteWorkflowRunResponse, CuratedDocumentUpdateRequest, CurationFieldActionRequest (+23 more)

### Community 24 - "Community 24"
Cohesion: 0.13
Nodes (26): ExtractionFileSummary, ExtractionOverview, ExtractionOverviewStatus, PromptTokenBudgeter, build_evidence_context_prompt(), build_evidence_context_prompt_components(), build_evidence_system_prompt_components_with_overview(), build_evidence_system_prompt_with_overview() (+18 more)

### Community 26 - "Community 26"
Cohesion: 0.12
Nodes (11): ContentChunk, DataPackage, DataSourceService, TaskStatus, DataSourceServiceTests, FakeBlobRepository, FakeOllamaClient, FakeSettings (+3 more)

### Community 28 - "Community 28"
Cohesion: 0.11
Nodes (24): defaultValueForSchema(), formatPrimitivePreview(), formatTreeNodeLabel(), getValueAtPath(), isPathProtected(), isRecord(), isTopLevelPath(), itemSchema() (+16 more)

### Community 29 - "Community 29"
Cohesion: 0.06
Nodes (26): ContentChunk, DataPackage, BytesIO, ContentChunk, DataPackage, FileEntry, OllamaClientWrapper, Settings (+18 more)

### Community 30 - "Community 30"
Cohesion: 0.14
Nodes (6): TestClient, FakeOllamaClient, FakeRemoteSettings, FakeRunningModel, FakeSettings, TestSystemApi

### Community 31 - "Community 31"
Cohesion: 0.15
Nodes (25): DataSourceService, UploadFile, DataPackage, ChunkingRequest, Tunable thresholds for the text-quality classifier.      All defaults mirror t, TextQualityConfig, description, Query (+17 more)

### Community 32 - "Community 32"
Cohesion: 0.16
Nodes (24): Any, EvidenceNote, EvidenceNote, _branch_category_affinities(), build_schema_branch_index(), build_schema_search_query(), _class_description(), _dedupe_branches() (+16 more)

### Community 33 - "Community 33"
Cohesion: 0.15
Nodes (5): EvidenceContext, CompactVocabResource, make_chunk(), make_service(), VocabQueryResult

### Community 34 - "Community 34"
Cohesion: 0.22
Nodes (26): Any, Exception, OllamaClientWrapper, Settings, apply_runtime_config(), available_model_summary(), clean_model_name(), diagnose_ollama_runtime() (+18 more)

### Community 35 - "Community 35"
Cohesion: 0.14
Nodes (4): DataPackage, FileEntry, ExtractionServiceWorkflowTests, WhitespaceTokenizer

### Community 36 - "Community 36"
Cohesion: 0.13
Nodes (19): ExtractionRunProgress, TaskStatus, ExtractionContext, QuantitativeAttribute, TaskStatus, CompleteWorkflowProgressResponse, CompleteWorkflowRunResponse, CuratedDocumentUpdateRequest (+11 more)

### Community 37 - "Community 37"
Cohesion: 0.20
Nodes (23): Any, EvidenceContext, EvidenceNote, ExtractionNormalization, BaseModel, InitialVocab, FileInventoryItem, A traced extraction object pairs an extracted class instance with the specific t (+15 more)

### Community 38 - "Community 38"
Cohesion: 0.17
Nodes (8): ContentChunk, ExtractionFileSummary, ExtractionOverview, PromptTokenBudgeter, RankedFile, ExtractionOverviewFilePreview, ExtractionVocabQueryConfig, FileRankingResult

### Community 39 - "Community 39"
Cohesion: 0.12
Nodes (22): VocabQuery, VocabQueryResult, Form, HttpUrl, SemanticService, UploadFile, CompactVocabResourceResponse, _vocab_query_result_response() (+14 more)

### Community 40 - "Community 40"
Cohesion: 0.23
Nodes (23): FullTextIndexInfo, Graph, Logger, Settings, VectorIndexInfo, Any, CreateIndexRequest, BaseIndex (+15 more)

### Community 41 - "Community 41"
Cohesion: 0.20
Nodes (22): Any, Neo4jDriver, OllamaClientWrapper, Settings, Neo4jDriver, _check_neo4j(), _check_ollama_chat(), _check_ollama_embedding() (+14 more)

### Community 42 - "Community 42"
Cohesion: 0.15
Nodes (21): Any, detect_enrichable_fields(), export_document_to_jsonld(), _format_json_path(), generate_profile_artifacts(), GeneratedProfileArtifacts, InvalidProfileIdentifierError, JsonLdExportResult (+13 more)

### Community 44 - "Community 44"
Cohesion: 0.12
Nodes (15): classify_text_line(), _entropy(), _features(), _has_many_control_chars(), _looks_like_structured_text(), Text quality gate for filtering semantically meaningful lines.  This module is, Contains patterns like:     - key: value     - key = value     - Markdown hea, Unit tests for the tunable text-quality classifier. (+7 more)

### Community 46 - "Community 46"
Cohesion: 0.19
Nodes (7): BytesIO, DataPackage, FileEntry, FileEntryNotFoundError, InvalidDataPackageZipFileError, FileType, ZipFile

### Community 47 - "Community 47"
Cohesion: 0.11
Nodes (7): Graph, TraversalDirection, VocabGraphStatement, VocabResource, VocabSchemeInfo, VocabSearchCandidate, SemanticGraphRepository

### Community 48 - "Community 48"
Cohesion: 0.16
Nodes (13): checkVocabularyEmbeddings(), deleteVocabulary(), getVocabulary(), importVocabulary(), listVocabularies(), queryVocabulary(), SearchMode, VocabEmbeddingStatus (+5 more)

### Community 49 - "Community 49"
Cohesion: 0.27
Nodes (13): FileEntry, PromptTokenBudgeter, attach_cosine_distances(), BufferWindow, combine_lines(), ContentChunk, FilteredLine, Combine each line with its neighbours using a sliding buffer window.      Retu (+5 more)

### Community 50 - "Community 50"
Cohesion: 0.17
Nodes (16): RankedFile, build_extraction_file_summary_prompt(), build_extraction_overview_fallback_prompt(), build_extraction_overview_prompt(), build_extraction_overview_prompt_components(), _compact_path_tree_lines(), compact_seeded_overview_for_prompt(), ExtractionFileContentWindow (+8 more)

### Community 51 - "Community 51"
Cohesion: 0.06
Nodes (20): Any, CurationLedgerRecord, DraftValidationResult, EvidenceContext, ExtractionRunResult, ExtractionRunState, FieldCompletionLedgerRecord, FilteredEvidenceLedger (+12 more)

### Community 52 - "Community 52"
Cohesion: 0.22
Nodes (6): Any, GeneratedProfileArtifacts, Path, ProfileManifest, FileSystemProfileRepository, ProfileNotFoundError

### Community 53 - "Community 53"
Cohesion: 0.11
Nodes (18): compilerOptions, allowJs, allowSyntheticDefaultImports, esModuleInterop, forceConsistentCasingInFileNames, isolatedModules, jsx, lib (+10 more)

### Community 54 - "Community 54"
Cohesion: 0.31
Nodes (5): ExtractionRunState, ExtractionChunkResult, evidence_context(), overview_for_file(), resource_context()

### Community 55 - "Community 55"
Cohesion: 0.11
Nodes (17): dependencies, react, react-dom, devDependencies, @types/react, @types/react-dom, typescript, vite (+9 more)

### Community 57 - "Community 57"
Cohesion: 0.16
Nodes (8): FakeFileEntry, ProtectedLineIndicesTests, Unit tests for the chunking domain logic, especially protected_line_indices., Line indices outside the file bounds are silently ignored., Minimal stand-in for FileEntry that only provides extracted content., When protected_line_indices contains [0], a dropped header is retained., With protected_line_indices=None the header is treated as usual., If a protected line is already KEEP, it appears only once.

### Community 58 - "Community 58"
Cohesion: 0.15
Nodes (10): get_datasource_service(), get_extraction_service(), get_neo4j_driver(), get_ollama_client(), get_profile_service(), get_settings(), get_task_registry(), ExtractionProfileDomainTests (+2 more)

### Community 59 - "Community 59"
Cohesion: 0.23
Nodes (16): Exception, Form, HttpUrl, ProfileService, UploadFile, File, ProfileDocumentRequest, delete_profile() (+8 more)

### Community 60 - "Community 60"
Cohesion: 0.29
Nodes (12): Any, build_file_ranking_prompt(), _contains_any(), fallback_file_ranking(), _fallback_reasons(), _fallback_score(), _file_extension(), FileContext (+4 more)

### Community 61 - "Community 61"
Cohesion: 0.21
Nodes (5): TestClient, FastAPI, FakeSemanticService, make_test_client(), SemanticApiRoutingTests

### Community 63 - "Community 63"
Cohesion: 0.16
Nodes (8): AgenticEntity, DataGeneratingActivity, EvaluatedEntity, Method, An experimental, measurement, acquisition, or processing activity that produces, A method, plan, protocol, pulse sequence, acquisition procedure, processing rout, The actual target entity evaluated by a data-generating activity, such as a samp, An entity with agency that can perform activities, such as a person, organizatio

### Community 64 - "Community 64"
Cohesion: 0.16
Nodes (10): EntityTypeNotSupportedError, IndexDuplicateError, IndexError, IndexNotFoundError, IndexTypeError, Base exception for index operations., Raised when an index cannot be found by name., Raised when multiple indexes share the same name. (+2 more)

### Community 65 - "Community 65"
Cohesion: 0.20
Nodes (13): JsonLdExportResult, ProfileManifest, ProfileValidationResult, ProfileValidationIssue, _jsonld_export_response(), JsonLdExportResponse, _profile_manifest_response(), _profile_validation_issue_response() (+5 more)

### Community 66 - "Community 66"
Cohesion: 0.18
Nodes (5): Any, GeneratedProfileArtifacts, ProfileManifest, ProfileRepository, ProfileRepository

### Community 67 - "Community 67"
Cohesion: 0.19
Nodes (5): ExtractionContext, ExtractionRunResult, model_output_for_file(), qualitative_context(), quantitative_context()

### Community 68 - "Community 68"
Cohesion: 0.26
Nodes (10): determine_file_type(), extract_text_from_file(), extract_text_from_pdf(), extract_text_from_sheets(), FileType, normalize_file_extension(), Extract text content from a file based on its type.      Dispatches to the app, Extract text content from a PDF file. (+2 more)

### Community 69 - "Community 69"
Cohesion: 0.36
Nodes (12): Path, doi_from_name(), field(), infer_key(), latex_clean(), main(), normalize(), normalize_arxiv() (+4 more)

### Community 70 - "Community 70"
Cohesion: 0.25
Nodes (7): import_initial_vocab(), main(), run_initial_vocab_bootstrap(), get_semantic_service(), lifespan(), Path, setup_logging()

### Community 71 - "Community 71"
Cohesion: 0.22
Nodes (3): JsonLdExportResult, ProfileManifest, ProfileValidationResult

### Community 73 - "Community 73"
Cohesion: 0.22
Nodes (4): Any, normalize_neo4j_value(), Convert Neo4j driver values into JSON-serializable Python values., Self

### Community 74 - "Community 74"
Cohesion: 0.25
Nodes (7): compilerOptions, allowSyntheticDefaultImports, composite, module, moduleResolution, skipLibCheck, include

### Community 75 - "Community 75"
Cohesion: 0.32
Nodes (8): decodeTraceEscapes(), findDirectRange(), findFlexibleWhitespaceRange(), findLineSequenceRange(), findTraceRange(), normalizeTraceSearchText(), stripTracePromptMetadata(), uniqueStrings()

### Community 76 - "Community 76"
Cohesion: 0.38
Nodes (4): DataPackageIdNotFoundError, DataPackageZipNotFoundError, InvalidDataPackageFileNameError, MultipleDataPackageZipFilesError

### Community 77 - "Community 77"
Cohesion: 0.40
Nodes (5): TaskInfo, _serialize_task(), TaskResponse, get_tasks(), Return all currently registered background tasks.

### Community 79 - "Community 79"
Cohesion: 0.33
Nodes (4): GenerateResponse, _nanoseconds_to_milliseconds(), Token usage tracking for structured completions., Create RunUsage from an ollama GenerateResponse object.

### Community 80 - "Community 80"
Cohesion: 0.33
Nodes (4): AsyncClient, Logger, Settings, The underlying ollama.AsyncClient for direct /api/generate calls.          Use

### Community 87 - "Community 87"
Cohesion: 0.83
Nodes (3): compact_result(), load_api_key(), main()

## Knowledge Gaps
- **132 isolated node(s):** `ProfileManifest`, `ProfileValidationResult`, `ProfileValidationIssue`, `JsonLdExportResult`, `Path` (+127 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **27 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BaseModel` connect `Community 37` to `Community 2`, `Community 4`, `Community 5`, `Community 6`, `Community 7`, `Community 13`, `Community 17`, `Community 18`, `Community 24`, `Community 31`, `Community 32`, `Community 33`, `Community 36`, `Community 39`, `Community 40`, `Community 41`, `Community 42`, `Community 45`, `Community 46`, `Community 49`, `Community 50`, `Community 60`, `Community 65`, `Community 77`?**
  _High betweenness centrality (0.157) - this node is a cross-community bridge._
- **Why does `ExtractionService` connect `Community 9` to `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 12`, `Community 14`, `Community 18`, `Community 20`, `Community 23`, `Community 25`, `Community 27`, `Community 32`, `Community 33`, `Community 35`, `Community 36`, `Community 38`, `Community 43`, `Community 45`, `Community 51`, `Community 54`, `Community 60`, `Community 62`, `Community 67`?**
  _High betweenness centrality (0.142) - this node is a cross-community bridge._
- **Why does `OllamaClientWrapper` connect `Community 4` to `Community 0`, `Community 5`, `Community 9`, `Community 14`, `Community 16`, `Community 20`, `Community 22`, `Community 25`, `Community 29`, `Community 34`, `Community 35`, `Community 36`, `Community 37`, `Community 38`, `Community 41`, `Community 43`, `Community 51`, `Community 54`, `Community 80`, `Community 91`, `Community 92`?**
  _High betweenness centrality (0.130) - this node is a cross-community bridge._
- **Are the 64 inferred relationships involving `ExtractionService` (e.g. with `Any` and `DataSourceService`) actually correct?**
  _`ExtractionService` has 64 INFERRED edges - model-reasoned connections that need verification._
- **Are the 203 inferred relationships involving `TaskStatus` (e.g. with `InitialTasks` and `Any`) actually correct?**
  _`TaskStatus` has 203 INFERRED edges - model-reasoned connections that need verification._
- **Are the 157 inferred relationships involving `Settings` (e.g. with `InitialTasks` and `AsyncClient`) actually correct?**
  _`Settings` has 157 INFERRED edges - model-reasoned connections that need verification._
- **Are the 146 inferred relationships involving `OllamaClientWrapper` (e.g. with `InitialTasks` and `Any`) actually correct?**
  _`OllamaClientWrapper` has 146 INFERRED edges - model-reasoned connections that need verification._