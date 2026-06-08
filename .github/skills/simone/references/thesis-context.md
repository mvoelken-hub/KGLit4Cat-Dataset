# SIMONE Thesis Context

Use this reference when the user asks about thesis objectives, research questions, scope, contribution framing, chapter structure, or current writing priorities.

Primary local sources to re-check:

- `docs/thesis/assets/agent-generated-assets/thesis_simone_working_document.md`
- `docs/thesis/sections/*.tex`
- `docs/WORKFLOW.md`

## Core Topic

Project topic: LLM-supported semantic metadata extraction from catalysis-related research data packages.

Prototype: SIMONE, Semantic Inference Module for Ontology-driven Node Extraction.

The thesis develops and evaluates a workflow that converts heterogeneous catalysis-related dataset packages into structured, semantically enriched metadata documents.

## Central Research Question

How can large language models be integrated into a controlled workflow for extracting, normalizing, and projecting semantic metadata from heterogeneous catalysis research data packages?

## Supporting Research Questions

1. Which metadata-relevant entities and attributes can be extracted from heterogeneous dataset packages using a chunk-wise LLM workflow?
2. How can extraction outputs be made traceable to file-level and text-level evidence?
3. How can extracted quantitative and qualitative attributes be normalized against controlled vocabularies such as QUDT, Voc4Cat, CHMO, and nmrCV?
4. How can intermediate extraction objects be projected into an application profile such as DCAT-AP Plus or ChemDCAT-AP-style metadata?
5. Which failure modes occur in extraction, grounding, and profile projection, and how can they be evaluated?

## Contribution Framing

Frame SIMONE as an assistance system, not an autonomous replacement for expert curation.

The contribution is the staged, inspectable architecture:

- ZIP-based data package ingestion.
- File-type-specific text extraction where possible.
- Semantic chunking.
- File ranking.
- Chunk-wise LLM extraction into an intermediate `ExtractionContext`.
- Traceability through file paths, chunk spans, and source-text snippets.
- Vocabulary-backed normalization for quantitative and qualitative attributes.
- Profile projection into schema-validated metadata.
- Frontend inspection and rerun support.

The strongest thesis claim is not that SIMONE perfectly extracts catalysis metadata. The stronger and more defensible claim is that a controlled staged workflow makes LLM-supported metadata extraction more traceable, inspectable, evaluable, and semantically interoperable than an unconstrained one-shot prompt.

## Scope

Included:

- ZIP-based research data packages.
- Supported text extraction from PDFs, spreadsheets/CSV, and text-like files.
- Semantic chunking based on embedding-distance breakpoints.
- Chunk-wise structured extraction.
- QUDT quantity-kind and unit normalization.
- Qualitative grounding against configurable RDF vocabularies such as Voc4Cat, CHMO, and nmrCV.
- Neo4j-backed hybrid vocabulary retrieval.
- Profile projection and schema validation.
- Frontend inspection of uploads, chunks, progress, extraction context, vocabulary queries, and final results.

Excluded or limited:

- Full scientific interpretation of every raw file format.
- Reliable image or binary extraction without text extraction.
- Replacement of domain-expert validation.
- Complete human-in-the-loop patch review.
- Exhaustive ontology engineering.
- Production-grade repository integration.

## Current Thesis Status

Developed:

- Conceptual direction and workflow narrative.
- Theoretical background: catalysis, metadata, Semantic Web, ontologies, LLMs, RAG, structured output.
- Method chapter foundation, because the prototype has a concrete architecture.

Underdeveloped:

- Evaluation chapter remains a scaffold.
- Manual reference annotations are not yet complete.
- Experimental results and baseline comparison still need to be produced.
- Discussion needs concrete failure modes and transferability analysis.

Recommended next writing work:

1. Finalize research questions and contribution statement.
2. Stabilize the method chapter against actual implementation.
3. Define evaluation dataset and annotation scheme.
4. Run extraction experiments and collect metrics.
5. Write evaluation around evidence tables.
6. Write discussion around limitations, failure modes, and transferability.

## Chapter Structure

Current intended structure:

1. Introduction.
2. Theoretical Background.
3. Current State of Research.
4. Methodical Approach.
5. Results and Evaluation.
6. Conclusion and Outlook.

Appendices should hold prompt templates, profile schema excerpts, example extraction contexts, vocabulary query examples, additional result tables, and an error-category catalog.
