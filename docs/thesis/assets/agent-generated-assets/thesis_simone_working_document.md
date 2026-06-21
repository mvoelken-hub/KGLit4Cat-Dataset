# Working Document: Thesis Work and SIMONE Extraction Prototype

**Project topic:** LLM-supported semantic metadata extraction from catalysis-related research data packages  
**Prototype:** SIMONE — Semantic Inference Module for Ontology-driven Node Extraction  
**Document type:** working overview for thesis writing, prototype documentation, and evaluation planning  
**Current focus:** staged extraction workflow, schema/profile construction, final semantic grounding, and evaluation design

---

## 1. Executive Summary

This thesis develops and evaluates an LLM-supported workflow for extracting semantic metadata from heterogeneous catalysis-related research data packages. The workflow addresses a practical research-data-management problem: experimental catalysis data are often stored in ZIP archives containing reports, spreadsheets, instrument exports, images, raw measurements, scripts, and descriptive notes. Much of the scientifically relevant metadata is not directly machine-actionable. It is distributed across files, expressed in inconsistent terminology, and often only interpretable by humans.

The SIMONE prototype implements a full-stack system that turns uploaded data packages into structured metadata proposals. It performs file handling, text extraction, file ranking, semantic chunking, chunk-wise LLM extraction, profile projection and reconstruction, validation, final vocabulary-backed normalization, progress tracking, and frontend-based inspection. The current implementation emphasizes evidence-backed profile construction followed by final semantic normalization.

The thesis should position SIMONE as an assistance system rather than an autonomous replacement for expert curation. Its contribution lies in decomposing metadata extraction into inspectable workflow stages, preserving provenance at chunk and source-text level, grounding extracted terms in controlled vocabularies, and validating the final result against an application profile.

---

## 2. Thesis Motivation

Catalysis experiments generate valuable data about materials, synthesis procedures, reactor setups, characterization methods, reaction conditions, analytical measurements, and performance indicators. These data are only reusable when the surrounding context is documented. A conversion value, selectivity, spectrum, chromatogram, or table is insufficient without information about the sample, catalyst history, reactor configuration, measurement method, units, operating conditions, and data-processing steps.

The FAIR principles require data to be findable, accessible, interoperable, and reusable. In catalysis, this is difficult because the data landscape is heterogeneous and often shaped by local laboratory practices. Metadata may appear in README files, Excel sheets, instrument exports, PDF reports, free-text descriptions, filenames, folder names, or embedded parameter sections. Manual extraction is possible, but it is slow, inconsistent, and difficult to scale.

LLMs provide a possible route for extracting structured candidate metadata from heterogeneous text. However, unrestricted LLM use is not sufficient for scientific metadata extraction. The workflow must constrain the model, preserve evidence, validate outputs, ground terms in controlled vocabularies, and keep the process inspectable. The thesis therefore investigates a staged workflow in which LLMs are embedded into a controlled extraction and semantic-enrichment architecture.

---

## 3. Research Objective

The central objective is to design, implement, and evaluate a workflow that converts heterogeneous catalysis-related data packages into structured, semantically enriched metadata documents.

A suitable central research question is:

> How can large language models be integrated into a controlled workflow for extracting, projecting, and semantically normalizing metadata from heterogeneous catalysis research data packages?

Supporting research questions:

1. Which metadata-relevant entities and attributes can be extracted from heterogeneous dataset packages using a chunk-wise LLM workflow?
2. How can extraction outputs be made traceable to file-level and text-level evidence?
3. How can intermediate extraction objects be projected into an application profile such as DCAT-AP Plus or ChemDCAT-AP-style metadata?
4. How can quantitative and qualitative attributes already placed in that profile be normalized against controlled vocabularies such as QUDT, Voc4Cat, CHMO, and nmrCV?
5. Which failure modes occur in extraction, profile construction, and final grounding, and how can they be evaluated?

---

## 4. Scope

### Included

- ZIP-based research data packages.
- Text extraction from supported file formats.
- Ranking files by expected metadata relevance.
- Semantic chunking based on embedding-distance breakpoints.
- Chunk-wise LLM extraction into an intermediate `ExtractionContext` representation.
- Quantitative attribute normalization against QUDT quantity kinds and units.
- Qualitative attribute grounding against configurable controlled vocabularies such as Voc4Cat, CHMO, and nmrCV.
- Neo4j-backed semantic retrieval with vector search, full-text search, reciprocal-rank fusion, and local graph expansion.
- Profile projection into a schema-validated metadata document.
- Frontend inspection of uploads, chunks, extraction progress, extraction context, vocabulary queries, and final results.

### Excluded or limited

- Full scientific interpretation of every raw file format.
- Reliable extraction from images or binary files unless text extraction is available.
- Replacement of domain-expert validation.
- Complete human-in-the-loop patch review, although review functions may be discussed as future work.
- Exhaustive ontology engineering for catalysis.
- Production-grade repository integration beyond prototype-level export/storage.

---

## 5. Conceptual Workflow

SIMONE uses a staged workflow rather than one large prompt. This is necessary because the model context would otherwise need to contain raw file content, task instructions, profile schema, vocabulary context, examples, and output constraints at the same time. The staged design reduces context pressure and makes the pipeline easier to inspect and evaluate.

### Stage 1: Data package upload and unpacking

The user uploads a ZIP-based data package. The backend stores the package, unpacks its contents, identifies file entries, records file paths, file names, extensions, and available byte sizes, and prepares the package for text extraction.

### Stage 2: Text extraction

For supported file formats, the system extracts textual content. Files that cannot be interpreted internally may still be retained as resources but are not directly processed for semantic extraction. This distinction is important: a binary file can still be part of the final dataset metadata even when its content is not semantically parsed.

### Stage 3: Semantic chunking

Long extracted text is split into semantically coherent chunks. The current prototype filters text lines, combines neighboring lines into buffered windows, embeds those windows, computes cosine distances between adjacent windows, and treats high-distance points as potential breakpoints.

The purpose is not only to reduce prompt size. It is also to avoid cutting through meaningful parameter groups, method sections, sample descriptions, or instrument-configuration blocks. A semantic chunk should represent a local unit of evidence that can be interpreted independently while preserving enough context for extraction.

### Stage 4: File ranking

The system ranks files by their expected relevance for metadata extraction. README files, metadata tables, manifests, reports, protocols, notebooks, scripts, and compact processed tables are prioritized. Raw binaries, large images, caches, archives, and files that appear to contain only numeric measurements are deprioritized.

This stage improves efficiency and extraction quality because context-rich files are processed first. It also enables early progress reporting and helps avoid wasting model calls on low-value files.

### Stage 5: Chunk-wise extraction

Each chunk is sent to an LLM with a structured-output prompt. The model produces an intermediate `ExtractionContext`, not the final metadata document. This intermediate representation contains traced extraction objects:

- `DataGeneratingActivity`: measurement, acquisition, analysis, processing, or generation run.
- `Method`: protocol, plan, pulse sequence, acquisition procedure, processing routine, or instrument procedure.
- `EvaluatedEntity`: sample, material, catalyst, specimen, model, or other evaluated target.
- `AgenticEntity`: person, organization, instrument, software system, or other actor.
- `Resource`: file, dataset, spectrum, table, image, report, checksum, or other data artifact.

Attributes are attached to the nearest meaningful object:

- `QuantitativeAttribute`: value, unit, and quantity kind.
- `QualitativeAttribute`: title and literal value.

Every extracted object is paired with a source-text snippet to preserve traceability.

### Stage 6: Merging and deduplication

Chunk-level extraction contexts are merged into a data-package-level context. Duplicate or near-duplicate objects are consolidated. The merged context becomes the basis for profile projection.

This separation helps identify whether an error originated in chunking, extraction, merging, profile construction, or final grounding.

### Stage 7: Profile projection and semantic reconstruction

The merged evidence context is projected into a selected application profile. The profile defines where extracted information belongs. Coverage patching and semantic reconstruction improve the draft before its fields are vocabulary-grounded.

The projected metadata draft is validated against the registered profile schema. Optional curation can update the draft before final grounding.

### Stage 8: Vocabulary-backed normalization

Quantitative attributes are normalized against QUDT:

- quantity kinds are queried against QUDT quantity-kind resources;
- units are queried against QUDT unit resources.

Qualitative attributes are queried against configurable vocabularies such as:

- Voc4Cat for catalysis terminology;
- CHMO for chemical methods;
- nmrCV for NMR terminology.

The semantic service retrieves candidate terms using a hybrid strategy:

1. vector similarity search over embedded vocabulary labels and descriptions;
2. full-text search over indexed labels, titles, symbols, and related lexical properties;
3. reciprocal-rank fusion to combine vector and lexical rankings;
4. local graph expansion around candidate seed nodes;
5. compaction into a candidate context for inspection or LLM-assisted selection.

The model then selects a candidate URI only when it clearly matches a value already placed in the finalized profile draft. Otherwise the mapping remains unresolved. This is preferable to forcing a possibly wrong URI. This distinction is central:

- a vocabulary answers what a concept, unit, method, or term means;
- a profile has already defined where that information belongs in the metadata document.

Vocabulary grounding is the final enrichment stage before final validation and result persistence.

### Stage 9: Inspection and reruns

The frontend supports inspection of uploaded packages, extracted content, chunks, progress, extraction contexts, vocabulary query settings, vocabulary query results, and final metadata output. Vocabulary queries can be reconfigured and rerun without repeating the entire extraction.

---

## 6. Prototype Architecture

### Backend

The backend is a Python/FastAPI application. Its core responsibilities are:

- data package management;
- text extraction and chunking;
- extraction workflow orchestration;
- communication with Ollama models;
- structured-output parsing and repair;
- vocabulary query handling;
- profile loading and validation;
- persistence of intermediate and final extraction artifacts;
- progress and token-usage reporting.

### Frontend

The frontend is a React/TypeScript application. Its current role is to make the workflow observable and usable:

- upload dataset packages;
- browse package contents;
- inspect file text and chunks;
- start and monitor extraction runs;
- inspect intermediate extraction context;
- configure vocabulary query parameters;
- rerun vocabulary queries;
- inspect normalized metadata and final outputs.

### Infrastructure

The prototype uses:

- Docker / Docker Compose for local full-stack operation;
- Neo4j for vocabulary graph storage and semantic retrieval;
- Ollama for local or remote embedding and chat model calls;
- a runtime directory for uploads, vocabularies, profiles, outputs, logs, and temporary artifacts.

The default embedding model is used for semantic search and chunking. The default chat model is used for extraction, candidate selection, fallback query generation, and profile projection.

---

## 7. Main Domain Objects

### `QuantitativeAttribute`

Represents a measured, calculated, or configured quantity.

Typical fields:

- identifier;
- value;
- unit;
- quantity kind.

Examples:

- temperature = 300 K;
- pressure = 1 bar;
- flow rate = 20 mL/min;
- pulse length = 10 us.

### `QualitativeAttribute`

Represents a qualitative value, setting, label, mode, material descriptor, or classification.

Examples:

- acquisition mode = single pulse;
- solvent = water;
- method type = NMR spectroscopy;
- sample state = spent catalyst.

### `DataGeneratingActivity`

Represents an activity that produces data.

Examples:

- NMR acquisition;
- catalytic reaction test;
- temperature-programmed reduction;
- GC analysis;
- data processing run.

### `Method`

Represents a method, protocol, procedure, or processing routine.

Examples:

- pulse sequence;
- reactor operation protocol;
- chromatographic method;
- baseline correction routine.

### `EvaluatedEntity`

Represents the entity being evaluated.

Examples:

- catalyst sample;
- material batch;
- reaction mixture;
- specimen.

### `AgenticEntity`

Represents an actor or system that performs or controls an activity.

Examples:

- instrument;
- software;
- organization;
- person;
- automated measurement system.

### `Resource`

Represents a data artifact.

Examples:

- uploaded ZIP package;
- CSV file;
- spectrum;
- report;
- processed table;
- image file.

---

## 8. Evaluation Plan

The evaluation chapter should be evidence-driven and should assess the workflow at multiple levels rather than only reporting whether the final metadata document validates.

### 8.1 Evaluation material

Select a small but representative set of catalysis-related data packages. The evaluation set should include:

- packages with clear README or metadata files;
- packages with mixed text and tabular files;
- packages with instrument exports;
- packages with raw or binary files that should be retained as resources but not parsed;
- at least one package with ambiguous terminology;
- at least one package where vocabulary grounding is expected to be difficult.

### 8.2 Reference annotation

Create a manually curated reference for selected packages. The reference should contain:

- expected data-generating activities;
- expected methods;
- expected evaluated entities;
- expected resources;
- key quantitative attributes;
- key qualitative attributes;
- expected or acceptable vocabulary URIs where possible;
- final profile fields.

The annotation does not need to cover every possible detail. It should focus on fields that are important for dataset reuse.

### 8.3 Metrics

Possible metrics:

- file-ranking quality: whether relevant files appear in the top ranks;
- chunking usefulness: whether chunk boundaries preserve meaningful context;
- object extraction precision, recall, and F1;
- attribute extraction precision, recall, and F1;
- unit normalization success rate;
- quantity-kind normalization success rate;
- qualitative vocabulary grounding success rate;
- unresolved-but-correctly-unresolved rate;
- schema validation success rate;
- number and type of warnings;
- token usage per package and per stage;
- runtime per package and per stage.

### 8.4 Qualitative error categories

Useful error categories:

- missed relevant file;
- poor text extraction;
- bad chunk boundary;
- over-extraction of technical header fields;
- under-extraction of important context;
- wrong object type;
- incorrect attachment of attribute to object;
- duplicate objects not merged;
- wrong quantity kind;
- wrong unit;
- forced vocabulary mapping where no candidate fits;
- unresolved mapping where a good candidate exists;
- profile projection omission;
- schema-valid but semantically wrong output;
- hallucinated object or unsupported relation.

### 8.5 Baseline options

Possible baselines:

1. manual extraction from the same packages;
2. single-prompt extraction without semantic chunking;
3. extraction without vocabulary grounding;
4. extraction with fixed-length chunks instead of semantic chunks;
5. lexical vocabulary search without vector retrieval.

A full comparison with all baselines may be too time-consuming. A minimal but useful evaluation would compare the staged SIMONE workflow against either manual extraction or a simpler single-prompt baseline.

---

## 9. Current Thesis Status

### Developed parts

The thesis already has a clear conceptual direction and a strong workflow narrative:

1. Catalysis data require contextual metadata to become reusable.
2. Semantic metadata and controlled vocabularies are necessary for interoperability.
3. Manual extraction does not scale.
4. LLMs can assist extraction but must be embedded into a controlled workflow.
5. SIMONE implements such a workflow through staged extraction, profile construction, and final vocabulary normalization.

The theoretical background is comparatively mature. It covers catalysis, metadata, Semantic Web technologies, ontologies, LLMs, retrieval-augmented generation, and structured-output techniques. The method chapter is also well developed because the current prototype has a concrete architecture that can be described in detail.

### Underdeveloped parts

The evaluation and discussion sections still need experimental results. These should become the main remaining thesis work. The introduction also needs to be converted from a structured scaffold into a polished chapter with clear research questions and contribution statements.

### Recommended next writing tasks

1. Finalize research questions and scope in the introduction.
2. Add a short thesis-contribution paragraph.
3. Convert the method chapter into a stable description of the implemented workflow.
4. Define the evaluation dataset and annotation scheme.
5. Run extraction experiments and collect metrics.
6. Write the evaluation chapter around evidence tables.
7. Write a discussion focused on limitations, failure modes, and transferability.
8. Keep implementation details that are not essential to the argument in the appendix.

---

## 10. Suggested Thesis Chapter Structure

A compact structure aligned with the current repository state:

1. **Introduction**
   - motivation;
   - problem statement;
   - research questions;
   - objectives and scope;
   - contributions;
   - thesis outline.

2. **Theoretical Background**
   - catalysis and catalytic experiments;
   - metadata in experimental chemistry;
   - Semantic Web technologies;
   - ontologies and controlled vocabularies;
   - LLMs and structured output;
   - retrieval-augmented generation.

3. **Current State of Research**
   - FAIR catalysis data infrastructures;
   - catalysis metadata profiles and vocabularies;
   - knowledge graphs in chemistry and catalysis;
   - LLM-based information extraction;
   - research gap.

4. **Methodical Approach**
   - workflow requirements;
   - staged architecture;
   - data package processing;
   - file ranking;
   - semantic chunking;
   - extraction context design;
   - profile projection and semantic reconstruction;
   - final vocabulary grounding;
   - validation and traceability.

5. **Results and Evaluation**
   - evaluation setup;
   - dataset sample;
   - reference annotation;
   - metrics;
   - extraction results;
   - vocabulary-grounding results;
   - profile-validation results;
   - qualitative error analysis;
   - comparison with baseline/manual process.

6. **Conclusion and Outlook**
   - summary of contributions;
   - answers to research questions;
   - limitations;
   - future work.

Appendices:

- prompt templates;
- profile schema excerpts;
- example extraction contexts;
- vocabulary query examples;
- additional result tables;
- error-category catalog.

---

## 11. Prototype Strengths

- Clear staged architecture.
- Intermediate representation instead of direct final-schema generation.
- Traceability through source-text snippets.
- Semantic chunking instead of purely fixed-size chunking.
- Vocabulary grounding through both vector and lexical retrieval.
- Reciprocal-rank fusion for candidate ranking.
- Local graph context expansion around candidate terms.
- Separation between profile construction and final vocabulary grounding.
- Schema validation of final output.
- Frontend support for inspection and reruns.
- Token and progress tracking.
- Modular design that supports evaluation by stage.

---

## 12. Prototype Limitations

- Extraction quality depends on text extraction quality.
- Binary, image, and raw instrument files may not be semantically interpreted.
- LLM output can still be schema-valid but semantically wrong.
- Vocabulary grounding can fail when terms are ambiguous or absent from the configured vocabularies.
- Candidate selection may still require expert review.
- Semantic chunking can produce poor boundaries when text is sparse or repetitive.
- Current frontend review functionality is not yet a complete manual patch-review workflow.
- Evaluation data and manually curated references still need to be created.
- Model choice, context length, and Ollama configuration affect reproducibility.

---

## 13. Discussion Points for the Thesis

### 13.1 Why a staged workflow matters

A staged workflow makes the system inspectable. Instead of asking whether an opaque LLM output is correct, the thesis can analyze where correctness is gained or lost: file ranking, chunking, extraction, merging, profile construction, or final vocabulary grounding.

### 13.2 Why an intermediate extraction context matters

The intermediate context prevents premature coupling to one metadata profile. This makes the extraction process more reusable. If the target profile changes, the extraction stage does not need to be redesigned completely.

### 13.3 Why vocabulary grounding matters

Structured JSON alone is not semantic metadata. A field labelled `temperature` or `NMR` is still ambiguous unless it is linked to a stable concept or unit identifier. Grounding improves interoperability and supports later querying, validation, and integration.

### 13.4 Why validation is necessary but insufficient

Schema validation ensures that the final document is processable, but it does not guarantee scientific correctness. A schema-valid metadata document can still contain wrong mappings, unsupported claims, or omitted context. The evaluation must therefore include semantic and qualitative assessment.

### 13.5 Why unresolved mappings can be acceptable

The workflow should not force a URI when no candidate clearly fits. A correct unresolved status is better than an incorrect semantic link. This should be treated explicitly in the evaluation.

---

## 14. Future Work

Potential future work after the thesis:

- implement a complete manual correction and patch-review workflow;
- add file-format-specific parsers for common instrument exports;
- integrate OCR or multimodal extraction for selected image/PDF cases;
- expand ontology and vocabulary coverage;
- add active-learning loops based on expert corrections;
- support batch extraction over repository-scale datasets;
- export RDF directly from the projected profile;
- integrate with Repo4Cat or related repository infrastructure;
- add SHACL-based validation in addition to JSON Schema or Pydantic validation;
- evaluate different LLMs and embedding models;
- compare semantic chunking with fixed-size and document-structure-aware chunking;
- improve reproducibility by storing model configuration, prompt version, and vocabulary version with every run.

---

## 15. Short Working Abstract

This thesis develops and evaluates SIMONE, a prototype workflow for LLM-supported semantic metadata extraction from heterogeneous catalysis research data packages. The workflow accepts ZIP-based data packages, extracts text from supported files, ranks metadata-relevant files, splits content into semantic chunks, and uses an LLM to generate traceable intermediate extraction contexts. This evidence is projected into a registered metadata profile, enriched through coverage patching and semantic reconstruction, and validated as a profile draft. In the final enrichment stage, quantitative fields can be normalized against QUDT quantity-kind and unit vocabularies, while qualitative fields can be grounded against domain vocabularies such as Voc4Cat, CHMO, and nmrCV. A Neo4j-backed semantic service retrieves vocabulary candidates through vector search, full-text search, reciprocal-rank fusion, and local graph expansion. The thesis investigates how this staged architecture can make LLM-supported metadata extraction more traceable, inspectable, and semantically interoperable for catalysis data management.
