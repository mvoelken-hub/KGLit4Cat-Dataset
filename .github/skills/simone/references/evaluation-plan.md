# SIMONE Evaluation Plan

Use this reference for evaluation design, experimental planning, metrics, baseline selection, and completeness-gap analysis.

Primary local sources to re-check:

- `docs/HANDOVER_complete_workflow_endpoint.md`
- `docs/PROTOTYPE_STATUS.md`
- `docs/evaluation/preliminary_evaluation.md`
- `docs/thesis/assets/agent-generated-assets/thesis_simone_working_document.md`
- `docs/thesis/sections/05_results_and_evaluation.tex`
- `data/datasets`
- `docs/WORKFLOW.md`

## Current Evaluation Status

The prototype supports evaluation by stage, but thesis-grade evaluation is not complete yet.

Available:

- Sample ZIP data packages under `data/datasets`.
- `POST /api/v1/extraction/workflows/complete` for upload, chunking, extraction, normalization, projection, validation, and result/progress URL creation.
- `simone evaluate run` and `simone evaluate score`.
- Focused manual reference annotations under `data/evaluation/references`.
- Workflow artifacts persisted in runtime output directories after extraction.
- Partial reports for missing, timed-out, or crashed runs.
- Manual-reference baseline summaries.
- Token usage and progress tracking.
- Unit tests for many backend components.

Missing or incomplete:

- Thesis-grade manual gold/reference annotations beyond the current focused references.
- Final evaluation tables based on multiple completed post-fix runs.
- Baseline runs.
- Broad error analysis backed by real extraction outputs.
- Reproducibility package describing model, prompt, vocabulary, profile, chunking settings, and dataset IDs.
- Runtime/scalability controls for verbose packages such as NMR exports.

## Evaluation Material

Select a small representative set of catalysis-related packages. Include:

- Packages with clear README or metadata files.
- Mixed text and tabular files.
- Instrument exports.
- Raw or binary files that should be retained as resources but not parsed.
- At least one ambiguous terminology case.
- At least one difficult vocabulary-grounding case.

Keep the evaluation small enough that manual reference annotation is realistic.

## Manual Reference Annotation

For each selected package, create a manually curated reference with:

- Expected data-generating activities.
- Expected methods.
- Expected evaluated entities.
- Expected resources.
- Key quantitative attributes.
- Key qualitative attributes.
- Expected or acceptable vocabulary URIs where possible.
- Expected final profile fields.

Do not try to annotate every possible detail. Focus on fields important for dataset reuse and on evidence needed to answer the research questions.

## Metrics

Useful quantitative metrics:

- File-ranking quality: relevant files in top ranks.
- Chunking usefulness: whether chunk boundaries preserve meaningful context.
- Object extraction precision, recall, and F1.
- Attribute extraction precision, recall, and F1.
- Unit normalization success rate.
- Quantity-kind normalization success rate.
- Qualitative vocabulary-grounding success rate.
- Correct unresolved mapping rate.
- Schema validation success rate.
- Number and type of warnings.
- Token usage per package and stage.
- Runtime per package and stage.

Use qualitative review for cases where exact precision/recall is too brittle.

## Error Categories

Track errors such as:

- Missed relevant file.
- Poor text extraction.
- Bad chunk boundary.
- Over-extraction of technical header fields.
- Under-extraction of important context.
- Wrong object type.
- Incorrect attribute attachment.
- Duplicate objects not merged.
- Wrong quantity kind.
- Wrong unit.
- Forced vocabulary mapping where no candidate fits.
- Unresolved mapping where a good candidate exists.
- Profile projection omission.
- Schema-valid but semantically wrong output.
- Hallucinated object or unsupported relation.

## Baseline Options

Possible baselines:

1. Manual extraction from the same packages.
2. Single-prompt extraction without semantic chunking.
3. Extraction without vocabulary grounding.
4. Fixed-length chunks instead of semantic chunks.
5. Lexical vocabulary search without vector retrieval.

For the thesis, a minimal useful comparison is staged SIMONE workflow versus either manual extraction or a simple single-prompt baseline. Avoid overextending the evaluation if time is short.

## Reproducibility Checklist

For each run, record:

- Dataset filename and package ID.
- Git commit or code snapshot.
- Profile identifier and profile version/source.
- Chat model, embedding model, context length, and Ollama runtime settings.
- Chunking settings.
- Selected qualitative vocabularies.
- Vocabulary versions/import sources where available.
- Prompt versions or prompt source file paths.
- Runtime, token usage, warnings, and extraction result path.

## Thesis-Critical Completeness

The current implementation is complete enough to support the thesis as a workflow prototype and has a repeatable evaluation harness. It is not complete enough to support strong quality claims until more completed runs and error analysis exist.

Current evidence to remember:

- `IR-IR.zip` completed through the automatic endpoint and produced a schema-valid profile document, but semantic quality scores were weak.
- `1H_NMR-1H_NMR.zip` previously timed out around `78/107` chunks, showing a runtime/scalability failure mode.
- Several reference packages still have partial or missing outputs.
- The manual-reference baseline is focused fact-level evidence, not a complete expert-curated metadata baseline.

Prioritize:

1. Repeatable extraction runs on the focused reference set.
2. Runtime controls for verbose packages.
3. Stronger semantic-grounding evidence.
4. A compact baseline.
5. Evidence tables and qualitative error analysis.
