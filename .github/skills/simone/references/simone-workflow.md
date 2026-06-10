# SIMONE Workflow Reference

Use this reference for workflow-stage questions, method-section accuracy, prototype/thesis alignment, and implementation boundary checks.

Primary local sources to re-check:

- `docs/WORKFLOW.md`
- `docs/thesis/assets/agent-generated-assets/thesis_simone_working_document.md`
- `backend/app/services/extraction_service.py`
- `backend/app/services/datasource_service.py`
- `backend/app/domain/datasources/chunking.py`

## Conceptual Workflow For Thesis Prose

The compact thesis-facing workflow is:

1. Data package upload and unpacking.
2. File-type-specific text extraction.
3. Semantic chunking and chunk persistence.
4. Extraction run initialization and profile/schema loading.
5. Deterministic file ranking.
6. Chunk ordering.
7. Chunk-wise LLM extraction into `ExtractionContext`.
8. Optional structured-output repair.
9. Incremental merging/context accumulation.
10. Vocabulary candidate discovery and query execution.
11. Vocabulary-backed normalization.
12. Profile projection.
13. Schema validation, result persistence, inspection, and reruns.

Do not describe file ranking as occurring before chunking in the current implementation. Chunking is a datasource operation and extraction refuses to start without completed chunks.

## Implementation Workflow

Implementation-level order:

1. Runtime preparation starts API, task registry, Neo4j/Ollama clients, semantic service, and route registration.
2. Dataset upload uses `POST /api/v1/datasources`.
3. `DataPackage.from_bytes()` validates ZIP input, recursively expands nested ZIPs, creates `FileEntry` objects, and computes a deterministic package ID.
4. Text extraction occurs when chunking reads each `FileEntry`.
5. `POST /api/v1/datasources/chunk` starts or returns datasource chunking.
6. `ContentChunk.create_chunks_for_file_entry()` extracts text, splits lines, filters text quality, reinserts protected lines, embeds line windows, calculates adjacent cosine distances, and stores chunks.
7. `POST /api/v1/extraction/run` verifies package/profile/schema and requires completed chunks.
8. Extraction task loads the selected profile, JSON Schema, validation schema, data package, and completed chunks.
9. Deterministic heuristic file ranking orders package file paths by expected metadata relevance.
10. Chunks are ordered by ranked file order, file path, and start line.
11. Each chunk is sent to the LLM to produce `ExtractionContext`.
12. Failed structured output can be queued for repair.
13. Completed chunk contexts are persisted and incrementally merged.
14. Vocabulary candidate discovery is scheduled for extracted quantitative and qualitative attributes.
15. Semantic service executes hybrid vocabulary queries using vector retrieval, full-text retrieval, reciprocal-rank fusion, and graph expansion.
16. Normalization selects QUDT quantity kinds/units and qualitative vocabulary mappings, preserving raw values with warnings when no mapping is selected.
17. Profile projection sends merged context, normalization results, warnings, and validation schema to the LLM.
18. The projected document is cleaned, validated against the registered profile, and persisted as `ExtractionRunResult`.
19. Results and token usage are retrieved through extraction endpoints and can be exported through profile APIs.

## Current Active Extraction Flow

The active backend workflow is a single resumable run, not the older initial-draft/patch-artifact pipeline.

Main extraction endpoints:

- `POST /api/v1/extraction/workflows/complete`
- `POST /api/v1/extraction/run`
- `GET /api/v1/extraction/run/{data_package_id}/progress`
- `POST /api/v1/extraction/run/{data_package_id}/pause`
- `PATCH /api/v1/extraction/run/{data_package_id}/vocab-query-config`
- `POST /api/v1/extraction/run/{data_package_id}/vocab-queries/rerun`
- `POST /api/v1/extraction/run/{data_package_id}/vocab-queries/{query_id}/rerun`
- `GET /api/v1/extraction/result/{data_package_id}`
- `GET /api/v1/extraction/{data_package_id}/token-usage`

Frontend compatibility names such as `extractInitialDraft`, `patchDraft`, and `PatchProgress` do not prove that backend patch artifact endpoints exist. Check current code before referring to patch review.

## Intermediate Representation

`ExtractionContext` stores traced extraction objects:

- `DataGeneratingActivity`
- `Method`
- `EvaluatedEntity`
- `AgenticEntity`
- `Resource`

Objects carry:

- `QuantitativeAttribute`: value, unit, quantity kind.
- `QualitativeAttribute`: title and literal value.
- `source_text`: exact evidence substring from the chunk.

The intermediate context is profile-independent. Profile projection happens only after extraction and normalization.

## Important Boundaries

- Images are not OCR-processed; they become placeholder text.
- Binary/raw instrument files may be retained as resources but are not semantically interpreted unless text extraction works.
- Manual patch review is not complete; older frontend review pieces are compatibility remnants.
- Extraction requires completed chunking.
- The complete workflow endpoint performs upload, chunking, extraction, normalization, projection, validation, and result/progress URL creation as one background workflow.
- Final output requires a registered profile and successful schema validation.
- JSON Schema validation is necessary but not sufficient for scientific correctness.
- Chunk-level extraction, candidate selection, fallback query generation, and projection depend on configured Ollama models. File ranking is deterministic by default.

## Thesis-Ready Interpretation

SIMONE is best described as a layered semantic extraction architecture:

```text
Raw files
  -> extracted text
  -> semantic chunks
  -> generic ExtractionContext
  -> vocabulary-normalized attributes
  -> profile-specific metadata document
  -> validated final output
```

The LLM is used at controlled points rather than as one unconstrained generator: chunk-level extraction, candidate selection/fallback query generation, and profile projection. File ranking is deterministic by default to keep evaluation stable and cheap.
