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
7. Ground selected terms against semantic vocabularies.
8. Project the accumulated context into a metadata profile.
9. Validate the final document against the selected profile.

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
- Quantitative evidence grouping remains domain-agnostic: candidates are grouped by numeric value, quantity label, optional unit, and source context rather than by hard-coded instrument- or domain-specific parameter names.

Thesis claim supported: SIMONE makes traceability and evidence grounding central to metadata construction.

### Accumulation Preserves Progressive Context

Each chunk contributes to an accumulated extraction context. The workflow progressively builds a dataset-level representation from local evidence units.

Reasoning:

- Metadata often depends on evidence distributed across files.
- Accumulation avoids treating chunks as isolated final answers.
- Intermediate state enables progress tracking, recovery, and inspection.
- Context caps are needed so accumulation remains usable by later prompts.

Thesis claim supported: SIMONE constructs metadata through staged intermediate representations.

### Vocabulary Grounding Separates Extraction From Semantics

After evidence extraction, selected terms are normalized against semantic vocabularies. This is a distinct step from detecting that a quantity, unit, material, method, or qualitative attribute exists.

Reasoning:

- LLM extraction identifies candidate meaning in source text.
- Vocabulary grounding links candidate meaning to reusable identifiers.
- Vector search, full-text search, graph context, and candidate selection provide a more controlled grounding process than free-text generation alone.
- Failed grounding should preserve raw values and warnings instead of fabricating semantic links.

Thesis claim supported: SIMONE combines LLM-assisted extraction with ontology- or vocabulary-backed semantic normalization.

### Profile Projection Is Late Binding

The workflow keeps a generic intermediate representation before projecting into a selected metadata profile. The final document is produced only after evidence extraction and vocabulary normalization.

Reasoning:

- A generic context can support different target schemas in principle.
- Profile-specific requirements should constrain final output, not all earlier evidence collection.
- Late projection reduces coupling between extraction prompts and one metadata profile.

Thesis claim supported: SIMONE separates extraction from profile-specific metadata generation.

### Validation Is A Boundary, Not Decoration

The final profile document must be validated. Validation marks the boundary between an extracted draft and a profile-conformant metadata artifact.

Reasoning:

- Schema validation exposes structural failure.
- Final metadata should be machine-checkable.
- Validation errors are evidence that extraction/projection did not satisfy the selected profile.
- A validated document is still not proof of scientific correctness; expert and benchmark evaluation remain separate.

Thesis claim supported: SIMONE constrains LLM output with explicit profile validation.

## Workflow Summary

```text
Dataset package
  -> text-accessible source artifacts
  -> initial dataset orientation
  -> source-local chunks
  -> grounded evidence candidates
  -> routed and accumulated evidence context
  -> vocabulary-normalized attributes
  -> profile-specific metadata document
  -> schema-validated result
```

## Current Thesis Boundaries

These boundaries belong in the thesis framing, regardless of implementation details:

- The workflow supports text-accessible heterogeneous files; image and binary interpretation remain limited unless text can be extracted.
- The workflow improves traceability but does not remove the need for expert review.
- Vocabulary grounding depends on vocabulary coverage, embeddings, graph retrieval, and candidate selection quality.
- Schema validation checks structure, not scientific truth.
- Evaluation claims require benchmark evidence and should not be inferred from architecture alone.
