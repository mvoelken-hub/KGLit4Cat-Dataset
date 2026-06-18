# SIMONE Prototype Status Compared With Teaser Claims

This note compares the public-facing prototype claims in the README, workflow notes, and thesis working material with the current implementation.

| Claim or teaser wording | Actual prototype status |
| --- | --- |
| SIMONE turns heterogeneous catalysis data packages into structured, reusable, FAIR-oriented metadata. | Implemented as a prototype pipeline: ZIP upload, file extraction, semantic chunking, chunk-wise LLM extraction, vocabulary normalization, profile projection, schema validation, and filesystem persistence. Scientific correctness still requires expert evaluation. |
| The workflow extracts document content and derives context from source material. | Implemented for text-accessible files including plain text, CSV/spreadsheets, PDFs, and nested ZIP contents. Images and binary instrument files are retained as resources but are not semantically interpreted unless text extraction succeeds. |
| The system creates an initial metadata draft and refines it iteratively. | Partly legacy wording. The active backend no longer uses the old manual patch-review chain. It runs direct extraction into `ExtractionContext`, vocabulary normalization, and profile projection into the final document. Some frontend function names still reflect the older draft/patch vocabulary. |
| Vocabulary-backed enrichment grounds selected fields semantically. | Implemented through Neo4j-backed vocabulary import/search, vector and full-text retrieval, graph expansion, candidate selection, and normalization records for quantitative and qualitative attributes. Quality depends on imported vocabularies, generated embeddings, and the configured LLM. |
| The workflow is traceable and reviewable. | Implemented at prototype level through chunk results, ranked files, extraction state, warnings, token usage, source-text evidence on extracted objects, and persisted artifacts under the runtime output directory. |
| A user can run the workflow from start to finish. | Stepwise UI/API execution already existed, but `POST /api/v1/extraction/run` required completed chunks. A new automatic endpoint, `POST /api/v1/extraction/workflows/complete`, now uploads a ZIP package, starts chunking, waits for chunking, starts extraction, and lets the backend carry the workflow to a final result. |
| The prototype is fully autonomous. | Not claimed as production autonomy. It still depends on registered profiles, reachable Neo4j/Ollama services, imported vocabularies, model authorization for the configured chat model, and successful schema validation. |
| The thesis can claim extraction quality and semantic grounding quality. | Not yet substantiated. A post-fix `IR-IR.zip` run completed and validated, but the preliminary scorer reported object macro F1 `0.2667`, attribute F1 `0.0`, and vocabulary mapping F1 `0.0`; `1H_NMR-1H_NMR.zip` exceeded a one-hour timeout at `78/107` chunks. |

## Automatic Workflow Endpoint

Use this endpoint after the API is running and at least one compatible profile is registered:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/extraction/workflows/complete" \
  -F "file=@./my-dataset.zip" \
  -F "profile_identifier=dcat-ap-plus"
```

Optional multipart fields:

- `qualitative_vocab_identifiers`: JSON string array or comma-separated identifiers.
- `buffer_window_size`: chunk buffer size, default `1`.
- `semantic_chunking_threshold`: chunking threshold from `0` to `100`, default `95`.
- `replace_existing_chunks`: replace chunks for a package with the same deterministic id, default `false`.
- `resume`: resume persisted extraction state, default `false`.
- `force_rerun`: clear persisted extraction artifacts before scheduling the workflow, default `false`.

For repeatable evaluation of deterministic package IDs, use `force_rerun=true` so older completed extraction artifacts do not mask current behavior.

The response includes:

- `data_package.id`
- `status`
- `progress_url`
- `result_url`

The workflow continues in background tasks. Poll the returned `progress_url` until the status is `completed`, then fetch `result_url`.
