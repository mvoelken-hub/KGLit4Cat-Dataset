# SIMONE Extraction Workflow Mental Model

> Snapshot note: This working document describes the current thesis-facing mental model of the SIMONE extraction workflow. Update it whenever the workflow interpretation, thesis claims, or reasoning behind the design changes.

This document explains the extraction workflow as a conceptual model for the thesis. It is not an implementation map. Code-specific endpoints, classes, services, and current prototype gaps belong in `PROTOTYPE_STATUS.md`; claim tracking belongs in `CLAIMS.md`.

## Thesis Role

SIMONE is framed as a workflow for turning heterogeneous catalysis research data packages into structured, reusable, FAIR-oriented metadata. The central thesis idea is not that an LLM directly writes final metadata in one step. The workflow instead constrains the task through staged representations, evidence grounding, vocabulary grounding, and profile validation.

The intended contribution is a traceable extraction architecture:

1. Ingest heterogeneous dataset archive.
2. Convert accessible source files into text.
3. Build dataset-level orientation context.
4. Split source text into manageable chunks.
5. Extract grounded evidence from chunks.
6. Accumulate and route extracted evidence.
7. Project the accumulated evidence context into a metadata profile draft.
8. Improve coverage, reconstruct semantic placement, and validate the draft. This completes the profile-draft stage.
9. Ground selected fields of the profile draft against semantic vocabularies into a separate final draft.
10. Validate and persist the grounded final document.

## Core Design Reasoning

### Heterogeneous Packages Need Staged Ingestion

Research datasets are rarely one clean document. They may contain PDFs, tables, text files, nested archives, images, raw instrument exports, scripts, and metadata fragments. SIMONE treats the uploaded package as a collection of source artifacts rather than a single prompt input.

Reasoning:

- A package-level workflow preserves file provenance.
- Text-accessible files can support extraction directly.
- Non-text or weakly interpretable files can still be represented as limitations or resources.
- The workflow can separate source availability from metadata confidence.

Thesis claim supported: SIMONE addresses heterogeneous dataset packages rather than only isolated text documents.

### Initial Context Supports Orientation, Not Evidence

Before chunk-level extraction, the workflow builds a coarse orientation over candidate files and dataset-level context. This helps later prompts understand likely dataset structure, but it is not treated as primary evidence.

Reasoning:

- Chunk prompts need enough context to avoid local misinterpretation.
- Large packages need a compact overview to manage attention.
- Per-file summaries describe file purpose and information nature in natural language instead of pre-sorting content into provenance, tool, setting, or numeric signal buckets.
- Evidence claims should still be grounded in source chunks, not in summary text.

Thesis claim supported: SIMONE separates orientation context from grounded source evidence.

### Chunking Is Context Management

Chunking exists because raw dataset text can exceed model context limits and mix unrelated concerns. The mental model is that chunks create bounded, source-local extraction units.

Reasoning:

- Smaller units keep LLM input manageable.
- Source-local chunks make traceability easier.
- Chunk ordering can prioritize likely metadata-rich files.
- Chunk boundaries should avoid both over-fragmentation and excessive context.

Thesis claim supported: SIMONE uses context management as an explicit workflow step rather than relying on a monolithic prompt.

### Extraction Produces Evidence Before Metadata

The chunk-level extraction stage should be understood as evidence gathering. The system asks for grounded observations and source text rather than immediately filling the target profile.

Reasoning:

- Evidence-first extraction reduces pressure to guess target fields too early.
- Source text makes later review and validation possible.
- Routing evidence into portable, contextual, and rejected groups separates strong claims from weak or local observations.
- Rejected/contextual evidence can expose uncertainty instead of silently disappearing.
- Evidence candidates carry both a broad signal category and a routing role. The category says what kind of source signal was observed; the role says how reusable metadata should treat it, such as identity, descriptor, context, qualitative attribute, or parameter.
- Evidence candidates preserve two provenance spans: an atomic copied `evidence_text` support span and a backend-derived copied `source_context` window that keeps nearby local scope such as section, block, resource, method, instrument, software, or activity context for later parent routing.
- Measurement-related evidence receives one structured semantic-routing call per note. Routes below `0.7` confidence, unavailable calls, unresolved ownership, and unsupported paths are recorded and skipped without rescue; accepted notes merge by target path plus stable semantic key before one activity/entity quantitative or qualitative attribute is appended per cluster. Low-level resource/distribution file listings stay out of the profile-draft flow.

Thesis claim supported: SIMONE makes traceability and evidence grounding central to metadata construction.

### Accumulation Preserves Progressive Context

Each chunk contributes to an accumulated extraction context. The workflow progressively builds a dataset-level representation from local evidence units.

Reasoning:

- Metadata often depends on evidence distributed across files.
- Accumulation avoids treating chunks as isolated final answers.
- Intermediate state enables progress tracking, recovery, and inspection.
- Context caps are needed so accumulation remains usable by later prompts.

Thesis claim supported: SIMONE constructs metadata through staged intermediate representations.

### Profile Projection Is Late Binding

The workflow keeps a generic evidence representation before projecting it into a selected metadata profile. Profile construction happens before vocabulary grounding so object and attribute placement can be assessed and corrected against the profile schema first.

Reasoning:

- A generic evidence context can support different target schemas in principle.
- Profile-specific requirements should constrain draft construction, not all earlier evidence collection.
- Coverage patching and semantic reconstruction establish which profile objects own each extracted value before semantic identifiers are assigned.
- Late profile binding reduces coupling between extraction prompts and one metadata profile.

Thesis claim supported: SIMONE separates extraction from profile-specific metadata generation.

### Vocabulary Grounding Is Final Enrichment

After profile draft construction, semantic reconstruction, and validation, selected fields already placed in the profile are normalized against semantic vocabularies. Vocabulary grounding is the final workflow enrichment step before final validation and persistence. It reads the raw reconstructed draft and writes a grounded final draft as a separate document.

Reasoning:

- Evidence extraction identifies candidate meaning in source text.
- Profile construction determines where that evidence belongs in the target schema.
- `generated_reconstructed_draft.json` is the completed profile-draft artifact and remains raw enough to inspect reconstruction output before grounding.
- `generated_final_draft` is the grounded final document produced from the reconstructed draft.
- Vocabulary grounding then links selected placed values to reusable identifiers.
- Fields typed as `DefinedTerm` are discovered from the active profile schema, not only from hardcoded field names.
- `type` fields represent vocabulary concepts, usually SKOS concepts. `rdf_type` fields represent ontology classes.
- Quantitative attributes receive deterministic QUDT class terms: the attribute itself is a QUDT `Quantity`, `has_quantity_type` terms are QUDT `QuantityKind`, and `unit` terms are QUDT `Unit`.
- Vector search, full-text search, graph context, and candidate selection provide a controlled grounding process.
- Failed grounding preserves raw values and warnings instead of fabricating semantic links.
- Grounding runs as its own stage (POST /extraction/stages/grounding/{id}/run) and never rebuilds the profile draft. It grounds the persisted reconstructed draft in place. Rerun all queries only refreshes existing query records without re-discovering fields.

Thesis claim supported: SIMONE combines evidence-backed profile construction with final ontology- or vocabulary-backed semantic normalization.

### Validation Is A Boundary, Not Decoration

The final profile document must be validated. Validation marks the boundary between an extracted draft and a profile-conformant metadata artifact.

Reasoning:

- Schema validation exposes structural failure.
- Final metadata should be machine-checkable.
- Validation errors are evidence that extraction/projection did not satisfy the selected profile.
- A validated document is still not proof of scientific correctness; expert and benchmark evaluation remain separate.

Thesis claim supported: SIMONE constrains LLM output with explicit profile validation.

### Requirement Reporting Separates Coverage, Semantics, And Trace

After initializing a dataset-level DCAT-AP+ draft shell, SIMONE builds a generic provenance core, checks coverage requirements and patches gaps from evidence, constructs attributes through parent-scoped semantic questions, runs semantic reconstruction triage, and then computes deterministic source trace scoring.

Reasoning:

- Coverage lists filled profile fields, including nested fields, as an inventory count rather than a quality percentage.
- Semantic requirements judge small, traceable semantic concerns such as title identity, description identity, real generation activity, technical-agent kind, method-plan presence, per-activity evaluation-target presence, duplicate attribute coherence, range decomposition, attribute label quality, attribute parent placement, and provenance-context placement.
- Semantic reconstruction reorganizes the current draft using requirement artifacts; it is not another evidence-search patching pass. Diagnosis and synthesis use separate structured-output contracts. Synthesis schemas are derived from the exact target path, while deterministic code compiles actions, creates containers, applies append/replace/remove/merge actions, validates each accepted action, and salvages independently valid actions. Mechanical merge/remove/move diagnoses cannot request synthesis, and attribute synthesis cannot replace a whole collection when only indexed entries were diagnosed.
- Duplicate attribute coherence and range decomposition run early as backend-compiled semantic repairs so later LLM diagnosis can focus on semantic placement instead of obvious cleanup. Same-parent numeric duplicates require the same canonical quantity and unit and tolerate harmless source-rounding differences; materially different values remain separate. Range repair can recover explicit bounds from the dataset description or compatible sibling boundary attributes, and parent placement removes verified cross-parent measurement duplicates from the less suitable owner.
- Evidence establishes whether a semantic requirement is applicable, but only values placed at its draft target paths can establish fulfillment. Mechanical draft-only checks omit evidence packets, while placement checks receive small requirement-specific packets.
- DataGeneratingActivity `evaluated_entity`/`evaluated_activity` state what that activity directly measured, observed, analysed, or studied. Generated outputs belong in `had_output_entity`; inputs belong in `had_input_entity` or `had_input_activity`. The active draft flow does not create Dataset `is_about_entity`/`is_about_activity` claims because they are easy to confuse with activity evaluation targets.
- Attribute parent placement is evaluated separately from attribute presence: each existing parent receives a narrow attribute question with the whole core draft as context, a focus target path/class, and parent-local evidence. The backend owns the schema path and does not expose its inferred parent-role label to the LLM. The LLM returns only structured quantitative/qualitative attribute intents. Forced profile rebuilds discard stale grounded and requirement-report state, and final deterministic guards rescore the report against the delivered grounded document.
- The profile draft artifacts expose the stage boundary explicitly: `generated_initial_draft.json`, `generated_core_draft.json`, `generated_attribute_draft.json`, and `generated_reconstructed_draft.json`.
- Source trace scoring summarizes source-file evidence quality for evidence that actually supports projected draft content.
- Dataset distribution material is no longer part of the profile-draft flow; profile construction now focuses on the draft itself and semantic attribute placement.
- Requirement and patch filters must stay domain-agnostic: no field names, vendor names, file names, instrument names, or benchmark examples may be hardcoded to improve a sample run.

Thesis claim supported: SIMONE distinguishes structural profile completion from semantic adequacy and evidence traceability.

## Workflow Summary

```text
Dataset package
  -> text-accessible source artifacts
  -> initial dataset orientation
  -> source-local chunks
  -> grounded evidence candidates
  -> routed and accumulated evidence context
  -> initialized profile draft
  -> generic provenance core construction
  -> parent-scoped attribute construction
  -> semantic reconstruction triage and validation
  -> completed raw reconstructed profile draft
  -> final vocabulary grounding of placed profile fields
  -> grounded final draft
  -> validated and persisted result
```

## Current Thesis Boundaries

These boundaries belong in the thesis framing, regardless of implementation details:

- The workflow supports text-accessible heterogeneous files; image and binary interpretation remain limited unless text can be extracted.
- The workflow improves traceability but does not remove the need for expert review.
- Vocabulary grounding depends on vocabulary coverage, embeddings, graph retrieval, and candidate selection quality.
- Schema validation checks structure, not scientific truth.
- Evaluation claims require benchmark evidence and should not be inferred from architecture alone.
