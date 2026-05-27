# Replace Pydantic AI with Custom Structured Completion Module

**Status:** Phase 2 replanned after extraction-domain simplification
**Created:** 2026-05-24  
**Last updated:** 2026-05-27

## Goal

Remove the `pydantic-ai` dependency and run all LLM interactions through the custom structured completion module in `app/ollama/`. Every call uses Ollama `/api/generate` with the `format` parameter, local JSON/Pydantic/schema validation, repair retries, and native token usage accounting.

The extraction domain has been rebuilt around one task-based workflow instead of the previous initial-context, initial-draft, patch, quality-review, and review-resolution agent chain.

## Current Extraction Workflow

The active extraction workflow is:

1. Rank files in the uploaded data package.
2. Iterate content chunks in ranked-file order.
3. Extract one `ExtractionContext` per chunk.
4. Merge and deduplicate extracted context objects after each chunk and persist the current interim `extraction_context.json`.
5. Normalize quantitative attributes with QUDT vocabulary queries and LLM candidate selection.
6. Normalize qualitative attributes against configured vocabularies with the same candidate-selection pattern.
7. Project the normalized aggregate context into the selected profile schema with a schema-bound LLM call.
8. Validate the projected document locally against the profile schema and persist the final result.

Chunking remains a prerequisite. Extraction does not auto-create chunks.

## Completion Inventory

| Completion | Output | Purpose |
|---|---|---|
| File ranking | `FileRankingResult` | Sort package files by likely metadata value. |
| Chunk extraction | `ExtractionContext` | Extract activities, entities, datasets, quantities, and qualitative attributes from one chunk. |
| Vocabulary candidate selection | `VocabularyCandidateSelection` | Choose a term URI from deterministic vocabulary query candidates, or return no match. |
| Vocabulary fallback query | `VocabularyFallbackQuery` | Create a focused alternate query when deterministic candidates do not fit. |
| Profile projection | Selected profile JSON Schema | Transform merged, normalized context into the final profile document. |

The profile transformation is intentionally LLM-based. A deterministic mapper from arbitrary `ExtractionContext` data into a profile-specific schema is too brittle for the current profiles. Deterministic code only orchestrates calls, builds vocabulary queries, records warnings/token usage, removes nulls, persists artifacts, and validates the final document.

## Public API

The old extraction endpoints were replaced by:

- `POST /api/v1/extraction/run`
- `GET /api/v1/extraction/run/{data_package_id}/progress`
- `GET /api/v1/extraction/result/{data_package_id}`
- `GET /api/v1/extraction/{data_package_id}/token-usage`

Progress responses include `progress.interim_context` when at least one chunk has been extracted. The same partial aggregate is written to `extraction_context.json` during the run so the UI can recover and display interim extraction results before profile projection completes.

Token usage buckets are:

- `file_ranking`
- `chunk_extraction`
- `quantity_vocab_selection`
- `qualitative_vocab_selection`
- `profile_projection`

## Current Breakages Addressed

The simplification removed legacy extraction artifacts and patch modules. The rebuild must not preserve stale imports from:

- `app.domain.extraction.artifacts`
- patch draft/discovery/repair modules
- patch quality modules
- patch review resolution modules
- `pydantic_ai.exceptions.AgentRunError`

The extraction package also needed package-relative imports; importing `app.domain.extraction` must not rely on top-level `file_ranking` or `extraction_context` modules.

## Structured Completion Architecture

`app/ollama/completion.py` is the only LLM structured-output surface:

```python
async def generate_structured(
    client: OllamaClientWrapper,
    *,
    model: str,
    system: str,
    prompt: str,
    output_type: type[BaseModel] | dict[str, Any],
    retries: int = 2,
    temperature: float = 0.0,
    seed: int = 42,
    think: bool | Literal["low", "medium", "high"] | None = False,
    num_ctx: int | None = None,
    keep_alive: float | str | None = -1,
    repair_model: str | None = None,
) -> CompletionResult[Any]: ...
```

Key behavior:

- Pydantic models provide JSON Schema through `model_json_schema()`.
- Raw dict `output_type` values are treated as JSON Schemas and validated locally.
- Parse/schema failures use narrow repair prompts.
- Empty model output fails immediately.
- Exhausted repair attempts raise `MaxRetriesExceeded`.
- Ollama token metadata is recorded through `RunUsage`.

## Dependency Cleanup

The backend app no longer needs Pydantic AI adapters. Final cleanup criteria:

```bash
rg "pydantic_ai|pydantic-ai" backend/app backend/tests
```

The command should return no app/test usage. Then remove `pydantic-ai-slim[openai]` from dependencies and refresh `uv.lock`.

## Test Plan

- Import smoke tests for `app.domain.extraction`, `app.services.extraction_service`, and `app.api.v1.extraction`.
- Unit tests for file ranking, prompt builders, context merge/deduplication, QUDT query builders, and qualitative query fan-out.
- Mocked `generate_structured()` tests for chunk extraction, candidate selection, fallback query, profile projection, token usage, and no-match behavior.
- Service tests for task lifecycle, chunk prerequisite, ranked chunk order, persisted interim context, persisted result, progress updates, and failed validation.
- Profile projection fixture test against `dcat-ap-plus`.
- API tests for run/progress/result/error mapping.
