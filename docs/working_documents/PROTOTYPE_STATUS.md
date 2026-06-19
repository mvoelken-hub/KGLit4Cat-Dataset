# SIMONE Prototype Implementation Status

> Snapshot note: This working document compares the current code implementation against the workflow mental model. Update it whenever implementation, workflow behavior, endpoints, artifacts, evaluation status, or thesis-facing claims change.

This document is the implementation-facing counterpart to `WORKFLOW.md`. It records what the current prototype actually implements, where it matches the mental model, and where gaps remain. The quick thesis claim ledger lives in `CLAIMS.md`.

## Current Implementation Compared With Mental Model

| Mental-model step | Current code implementation | Status / gap |
| --- | --- | --- |
| Ingest heterogeneous dataset archive | ZIP upload, deterministic package IDs, recursive nested ZIP expansion, file entries persisted through filesystem repositories. | Implemented for ZIP-based packages. |
| Convert accessible source files into text | Text extraction covers plain text/default decoded files, CSV/spreadsheets, PDFs, and placeholder text for images. | Implemented for text-accessible sources. Images are not OCR-processed; binary instrument files are not semantically interpreted unless text extraction succeeds. |
| Build dataset-level orientation context | Initial file summaries, an extraction overview, and a compact dataset summary are generated and stored in workflow state. Backend-filled file summary fields are omitted from the LLM-visible schema. | Implemented. Used as orientation context, not source evidence. |
| Split source text into manageable chunks | Chunking supports semantic embedding-distance breakpoints and fixed-token grouping, with min/max token post-processing. | Implemented. Stepwise extraction still requires completed chunks. |
| Extract grounded evidence from chunks | Chunk extraction uses structured LLM output for evidence candidates, validates copied evidence against chunk text, critiques evidence, and routes it into portable/contextual/rejected groups. Backend-filled evidence fields are omitted from the LLM-visible schema to save prompt/output tokens. | Implemented. Quality depends on configured chat model and source text quality. |
| Accumulate and route extracted evidence | Routed evidence is merged into interim extraction context and persisted with progress, warnings, and token usage. | Implemented. Context size still needs caps and careful prompt management. |
| Ground selected terms against vocabularies | Semantic service imports RDF vocabularies, creates vector/full-text indexes, retrieves candidates, expands graph context, and supports candidate selection for quantitative and qualitative attributes. | Implemented. Quality depends on vocabulary coverage, embeddings, Neo4j state, and LLM candidate selection. |
| Project accumulated context into metadata profile | Final profile projection uses selected profile schema and normalized context to produce a profile-shaped document. | Implemented. Projection can fail if schema requirements are not satisfied. |
| Validate final document | Result is validated against selected profile before final persistence. | Implemented. Validation checks schema conformance, not scientific correctness. |
| Retrieve result and inspect artifacts | Result, progress, warnings, token usage, vocabulary query records, and evidence/projection state are persisted and exposed through API/frontend paths. | Implemented at prototype level. UI/terminology still has some legacy draft/patch naming. |

## Current Workflow Entrypoints

Stepwise workflow:

1. Upload package.
2. Optionally run initial context generation.
3. Run chunking.
4. Run extraction with a registered profile.
5. Poll progress and fetch final result.

Complete workflow:

1. Upload ZIP package.
2. Run initial context generation.
3. Run chunking.
4. Run extraction.
5. Poll progress and fetch final result.

Important current API surfaces:

- `POST /api/v1/datasources`
- `POST /api/v1/datasources/chunk`
- `POST /api/v1/extraction/run/{data_package_id}/initial-context`
- `GET /api/v1/extraction/run/{data_package_id}/initial-context/progress`
- `POST /api/v1/extraction/run`
- `POST /api/v1/extraction/workflows/complete`
- `GET /api/v1/extraction/workflows/complete/{data_package_id}/progress`
- `GET /api/v1/extraction/result/{data_package_id}`
- `GET /api/v1/extraction/{data_package_id}/token-usage`

## Complete Workflow Options

The complete workflow endpoint currently accepts:

- `file`
- `profile_identifier`
- `qualitative_vocab_identifiers`
- `buffer_window_size`
- `semantic_chunking_threshold`
- `chunking_strategy`
- `fixed_tokens_per_chunk`
- `min_tokens_per_chunk`
- `max_tokens_per_chunk`
- `replace_existing_chunks`
- `resume`
- `force_rerun`

Use `force_rerun=true` for repeatable evaluation runs with deterministic package IDs so older artifacts do not mask current behavior.

## Current Prototype Boundaries

| Boundary | Current implementation status |
| --- | --- |
| Image understanding | Images return placeholder text; no OCR or visual interpretation is implemented. |
| Binary/instrument files | Retained as package resources, but not semantically interpreted unless text extraction succeeds. |
| Manual patch review | Old frontend patch-review concepts remain as compatibility stubs; active backend workflow is evidence extraction, normalization, projection, and validation. |
| Evaluation completeness | Early evaluation runs exist, but thesis-level quality claims are not fully substantiated yet. |
| Vocabulary coverage | Initial vocabularies can be imported, but grounding quality depends on imported vocabularies, term schemes, embeddings, and candidate selection. |
| Model dependency | Extraction, overview generation, candidate selection, fallback query generation, and profile projection depend on the configured Ollama chat model. |
| Profile dependency | Final output requires a registered compatible profile and successful schema validation. |
| Scientific correctness | A schema-valid final result is not automatically scientifically correct. Expert or benchmark evaluation is still required. |
| Quantitative attribute coverage | Requirement enrichment groups portable quantitative evidence by numeric value, quantity label, optional unit, and source neighborhood, then projects each group to a schema-valid owner or records a skip reason. Generic gates reject identifier/path/version noise and cap repeated row-like measurements while avoiding hard-coded instrument parameter whitelists. |

## Evaluation Status Snapshot

Current evidence is not enough to claim extraction or grounding quality broadly. A previous `IR-IR.zip` run completed and validated, but preliminary scoring reported low object/attribute/vocabulary F1. A `1H_NMR-1H_NMR.zip` run exceeded a one-hour timeout before completion. Keep quality claims conservative until the benchmark and evaluation tables are complete.
