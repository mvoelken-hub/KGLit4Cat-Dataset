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
| Extract grounded evidence from chunks | Chunk extraction uses structured LLM output for evidence candidates, validates copied evidence against chunk text, critiques evidence, and routes it into portable/contextual/rejected groups. Evidence categories separate resource, method, primary measurement, measurement condition, software, activity, instrument/device, surrounding metadata, and other signals. Each candidate carries a required routing role: qualitative attribute, identity, descriptor, context, parameter, or other metadata. Candidates also carry copied `evidence_text` plus a backend-derived copied `source_context` window so later profile stages can see local section/block scope without relying on paraphrased claims or extra extraction output. Backend-filled evidence fields are still overwritten by validation. | Implemented. Quality depends on configured chat model and source text quality; a small live probe showed the model can emit the new role field after prompt tightening. |
| Accumulate and route extracted evidence | Routed evidence is merged into interim extraction context and persisted with progress, warnings, and token usage. Exact and same-source semantic duplicate evidence notes are filtered while independent source support is retained. | Implemented. Context size still needs caps and careful prompt management. |
| Project accumulated evidence into metadata profile | Profile projection uses the selected schema to create an initial draft, improve evidence-backed coverage, evaluate semantic requirements, and reconstruct semantic placement. | Implemented. Projection can fail if schema requirements are not satisfied. |
| Validate and optionally curate profile draft | Draft writes are schema-validated throughout profile construction; `generated_reconstructed_draft.json` is the completed raw profile-draft artifact and can be curated before final grounding. | Implemented. Validation checks schema conformance, not scientific correctness. |
| Ground selected profile fields against vocabularies | After profile construction, the semantic service discovers schema-present `DefinedTerm` fields, applies role-level grounding policy, and writes selected terms into `generated_final_draft` from `curated_document` when present or from `generated_reconstructed_draft` otherwise. `type` is treated as a vocabulary concept role; `rdf_type` is treated as an ontology-class role. Quantitative attributes receive deterministic QUDT class terms for Quantity, QuantityKind, and Unit. | Implemented. Quality depends on vocabulary coverage, embeddings, Neo4j state, configured role policies, and LLM candidate selection. |
| Retrieve result and inspect artifacts | Result, progress, warnings, token usage, vocabulary query records, grounding policy snapshots, normalization records, validation results, and evidence/projection state are persisted and exposed through stage/workflow API and frontend paths. | Implemented at prototype level. Some internal artifact names still use draft/projection terminology where they describe profile construction. |

The backend extraction implementation is split by workflow responsibility: `WorkflowService` orchestrates task scheduling, progress, state, and result retrieval; stage services handle orientation, evidence extraction, profile projection, curation, and final vocabulary grounding. Deterministic extraction helpers remain in `app.domain.extraction` rather than in runtime services.

## Current Workflow Entrypoints

Stepwise workflow:

1. Upload package.
2. Optionally run initial context generation.
3. Run chunking.
4. Build the generated profile draft from the persisted evidence context with a registered profile.
5. Poll progress and fetch final result.

Complete workflow:

1. Upload ZIP package.
2. Run initial context generation.
3. Run chunking.
4. Run extraction.
5. Poll progress and fetch final result.

Within extraction, `target_stage="profile"` returns after profile construction and semantic reconstruction, with the raw reconstructed draft considered the completed profile-draft stage artifact. `target_stage="grounding"` and `target_stage="complete"` continue from the curated document when present, otherwise from that reconstructed draft, into vocabulary normalization and grounded final-draft persistence.

Important current API surfaces:

- `POST /api/v1/datasources`
- `POST /api/v1/datasources/chunk`
- `POST /api/v1/extraction/stages/orientation/{data_package_id}`
- `GET /api/v1/extraction/stages/orientation/{data_package_id}/progress`
- `POST /api/v1/extraction/stages/evidence`
- `GET /api/v1/extraction/stages/evidence/{data_package_id}/progress`
- `POST /api/v1/extraction/stages/profile`
- `PUT /api/v1/extraction/stages/curation/{data_package_id}/document`
- `POST /api/v1/extraction/stages/curation/{data_package_id}/field`
- `PATCH /api/v1/extraction/stages/grounding/{data_package_id}/config`
- `POST /api/v1/extraction/stages/grounding/{data_package_id}/rerun`
- `POST /api/v1/extraction/workflows`
- `GET /api/v1/extraction/workflows/{data_package_id}/progress`
- `GET /api/v1/extraction/workflows/{data_package_id}/token-usage`
- `GET /api/v1/extraction/results/{data_package_id}`

## Complete Workflow Options

The complete workflow endpoint currently accepts:

- `file`
- `profile_identifier`
- `qualitative_vocab_identifiers`
- role-level grounding policy for `type` and `rdf_type`
- `buffer_window_size`
- `semantic_chunking_threshold`
- `chunking_strategy`
- `fixed_tokens_per_chunk`
- `min_tokens_per_chunk`
- `max_tokens_per_chunk`
- `replace_existing_chunks`
- `resume`
- `force_rerun`

Use `force_rerun=true` for repeatable workflow runs with deterministic package IDs so older artifacts do not mask current behavior.

## Current Prototype Boundaries

| Boundary | Current implementation status |
| --- | --- |
| Image understanding | Images return placeholder text; no OCR or visual interpretation is implemented. |
| Binary/instrument files | Retained as package resources, but not semantically interpreted unless text extraction succeeds. |
| Manual patch review | Removed from the active API/frontend path; active backend workflow is evidence extraction, profile construction and semantic reconstruction, optional curation, final vocabulary grounding, validation, and persistence. |
| Evaluation completeness | In-repo offline scoring harness removed; thesis-level quality evaluation will be done later against completed workflow outputs. |
| Vocabulary coverage | Initial vocabularies can be imported, but grounding quality depends on imported vocabularies, term schemes, embeddings, role-level policy, and candidate selection. |
| Model dependency | Extraction, overview generation, candidate selection, fallback query generation, and profile projection depend on the configured Ollama chat model. |
| Generation determinism | Ollama runtime settings expose generation temperature and context-derived output-token cap enforcement. Default prototype behavior uses temperature 0 and enforced output caps for future LLM calls. |
| Profile dependency | Final output requires a registered compatible profile and successful schema validation. |
| Scientific correctness | A schema-valid final result is not automatically scientifically correct. Expert or benchmark evaluation is still required. |
| Quantitative attribute coverage | Requirement enrichment gives each routable measurement note one structured LLM call for activity/entity target path, stable merge key, and confidence. Routes below `0.7` confidence, unavailable calls, unresolved ownership, and unsupported paths are logged and skipped; no deterministic rescue path remains. Accepted notes merge by target path plus merge key and append one quantitative or qualitative attribute per cluster. |
| Requirement report scoring | `requirement_report.json` separates deterministic filled-field coverage inventory, small LLM semantic requirement assessment, semantics-driven reconstruction records, and deterministic used-evidence source trace scoring. Coverage patching is category-gated and no longer uses a single metadata completeness score. Omitted semantic assessments are recorded as `unanswered` rather than converted into missing facts. |
| Semantic reconstruction | After semantic requirement assessment, projection runs a semantics-driven draft-field iteration. Backend cleanup compiles same-parent duplicate merges, description/sibling-grounded range decomposition, and verified cross-parent duplicate removal. Rounded numeric variants merge only when canonical quantity and compatible units agree. LLM diagnosis and synthesis use separate contracts; synthesis schemas come from the exact profile target path. Collection-wide attribute replacement and synthesis-backed mechanical actions are rejected. Requirement-specific evidence limits keep mechanical checks draft-only and placement prompts small. Deterministic code validates each action, salvages valid actions, prevents evidence-only false fulfillment, and persists diagnosed defects, compiled actions, applied/rejected counts, and reasons. Defective requirements with no safe action are recorded as `unresolved`, not skipped. |
| Dataset-description mining | After initial draft creation, the profile stage extracts independently validated atomic facts from top-level dataset descriptions, adapts them into a local portable evidence context, and reuses requirement selection, semantic measurement routing, patch validation, and rollback. Descriptions remain unchanged during coverage patching; `description_facts.json` records validated and rejected facts, and description-derived evidence is excluded from source-trace scoring. | Implemented. Mining failure writes an audit artifact and falls back to source evidence. |
| Profile draft artifacts | Profile draft persistence stores `generated_initial_draft.json`, `generated_patched_draft.json`, and `generated_reconstructed_draft.json` so coverage patching and semantic reconstruction effects can be compared directly. The profile-draft stage is considered completed at `generated_reconstructed_draft.json`; grounding writes a separate `generated_final_draft` plus grounding artifacts. The draft flow no longer includes `dataset_distribution`. |
| Stage execution boundary | Evidence extraction and generated profile draft construction use separate stage entrypoints and task names. The profile stage consumes persisted evidence context and does not rerun chunk evidence extraction. Stage reruns clear only the relevant token-usage agents. |
| Evaluation defensibility | Projection filters must not hardcode dataset-specific keys, vendors, instruments, file names, or benchmark examples. Sample-specific fixes belong in prompt/category semantics or profile-declared rules before they can affect evaluation. |

## Domain-Specific Code To Generalize (Track For Later Sessions)

> Callout for future sessions: the code below hardcodes concepts tied to the current IR-spectroscopy evaluation dataset (wavenumber, transmittance, `1/cm`, point count, X/Y axis labels). This conflicts with the "Evaluation defensibility" boundary above ("Projection filters must not hardcode dataset-specific keys, vendors, instruments, file names, or benchmark examples"). It predates the 2026-06-22 grounding work and was NOT introduced by the grounding/normalization changes — it lives in the quantitative projection stage. Generalize it before making broad quality claims; do not let it silently bias evaluation.

Hardcoded dataset-specific logic in `backend/app/services/projection_service.py`:

| Location (function) | Lines | What it hardcodes | Generalization direction |
| --- | --- | --- | --- |
| `_attribute_parent_survivor_score` | ~1645, ~1674 | `pseudo_units = {"transmittance", "intensity", "count", "points", "point count"}`; family match on `("wavenumber", "transmittance", "point count", "resolution", "scaling")` | Drive pseudo-unit and family detection from the profile schema / declared measurement semantics, not a fixed spectroscopy set. |
| `_range_bounds_from_sibling_attributes` | ~1743-1744 | Synonym sets `"wavenumber": {"min wavenumber", "max wavenumber"}`, `"transmittance": {"min transmittance", "max transmittance"}` | Derive min/max sibling grouping from attribute titles/schema, not hardcoded axis quantities. |
| `_range_base_label` | ~1838-1841 | Axis-text → quantity: `"y"/"max y"/"min y"` → `transmittance`; `"x"/"max x"/"min x"/"first x"/"last x"` → `wavenumber` | Let the LLM/context identify the measured quantity for an axis label; do not map single letters to specific physical quantities. |
| `_range_unit_text` | ~1847 | Unit candidate list `("1/cm", "1 cm", "cm-1", "percent", "%")` | Use the vocabulary/units actually present in the profile or QUDT, not a fixed IR unit list. |
| `_normalized_attribute_label` | ~4212-4251 | `"data points" → "point count"`, `"transmittance value" → "transmittance"`, `"wavenumber value" → "wavenumber"`, and X/Y token sets → `max/min transmittance/wavenumber` | Normalize labels via schema-aware or LLM-driven disambiguation instead of a spectroscopy synonym map. |
| `_normalized_unit` | ~4261-4265 | `"1 cm"/"1/cm"/"cm-1" → "1/cm"`, `"percent" → "%"` | Normalize units through the grounding/vocabulary layer (unit selection) rather than a hardcoded IR unit map. |

Note for later sessions: the grounding stage (vocabulary query-formulation + candidate selection, 2026-06-22) is domain-agnostic by design — it feeds the document's own titles/descriptions as context and uses only domain-general prompt guidance. The remaining dataset coupling is the projection-stage code above, which should be generalized (or moved into profile-declared rules / vocabulary-backed normalization) so the whole pipeline is dataset-agnostic before any benchmark evaluation.

## Evaluation Status Snapshot

Current evidence is not enough to claim extraction or grounding quality broadly. Keep quality claims conservative until a fresh end-to-end evaluation pass is rebuilt outside the removed offline harness.
