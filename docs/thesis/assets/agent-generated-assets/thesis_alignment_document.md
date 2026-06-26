# Thesis Alignment Document

**Document type:** Working reference for maintaining thematic consistency across all thesis chapters  
**Created:** 2025-07-11  
**Last updated:** 2025-07-11  
**Status:** Active — consult before writing or revising any chapter  

---

## 1. Purpose of This Document

This document serves as a single point of reference that articulates the central theme of the thesis, maps every chapter to that theme, defines realistic goals for each chapter, and identifies the through-lines that must remain visible from the introduction to the conclusion. It is designed to be consulted whenever a chapter is written, revised, or restructured, so that the thesis reads as a coherent argument rather than a collection of independently written sections.

---

## 2. Central Theme

### 2.1 One-Sentence Statement

> This thesis develops and evaluates a controlled, staged LLM workflow — SIMONE — that converts heterogeneous catalysis research data packages into semantically grounded, schema-validated metadata by separating evidence extraction, profile construction, and vocabulary normalization into inspectable pipeline stages.

### 2.2 Expanded Theme

Catalysis experiments generate data whose scientific value depends on contextual metadata: sample provenance, catalyst history, reactor configuration, measurement methods, operating conditions, and processing steps. These metadata are currently locked in heterogeneous, semi-structured data packages — ZIP archives containing reports, spreadsheets, instrument exports, README files, scripts, and images — that are not machine-actionable. Manual extraction is slow, inconsistent, and does not scale.

Large language models can extract structured information from heterogeneous text, but unrestricted LLM use is insufficient for scientific metadata. The model may hallucinate, lose provenance, produce syntactically valid but semantically wrong output, or fail to ground terms in controlled vocabularies. A trustworthy workflow must therefore decompose the extraction task into stages that are individually inspectable, that preserve evidence at each step, and that ground the final output in community standards.

SIMONE addresses this need through a staged architecture: file handling and text extraction, semantic chunking, file ranking, chunk-wise evidence extraction into an intermediate representation, merging and deduplication, profile projection and semantic reconstruction, schema validation, and final vocabulary-backed normalization against QUDT, Voc4Cat, CHMO, and nmrCV. The system is positioned as an assistance tool that reduces manual curation burden while keeping a human expert in the loop for final review.

### 2.3 The Four Pillars This Thesis Connects

The work sits at the intersection of four domains identified in the literature wiki overview:

| Pillar | Role in This Thesis | Key Literature Anchors |
|--------|---------------------|------------------------|
| **Catalysis science** | The application domain. Catalytic experiments produce the heterogeneous data packages that SIMONE processes. Metadata must capture the full experimental chain from catalyst synthesis to performance testing. | `wulf2021unified`, `mendes2025data`, `marshall2023digital`, `moshantaf2024advancing` |
| **FAIR data infrastructure** | The target context. SIMONE's output is designed to be depositable in FAIR repositories and interoperable with NFDI4Cat/NFDI4Chem infrastructure. | `nfdi4cat2024whitepaper`, `kushnarenko2025repo4cat`, `schumann2025nomad`, `veluvali2025bridging` |
| **Semantic web & ontologies** | The representation layer. SIMONE uses RDF vocabularies, application profiles (ChemDCAT-AP), SHACL validation, and controlled vocabularies to produce machine-actionable metadata. | `stroemert2026chemdcat`, `w3c2017shacl`, `arp2015building`, `guarino1998formal` |
| **LLM-based extraction** | The technical method. SIMONE uses LLMs for evidence extraction, profile projection, and vocabulary candidate selection, embedded in a controlled pipeline with structured output and validation. | `dagdelen2024structured`, `schilling2025text`, `caufield2024spires`, `walls2025catminer` |

### 2.4 What Distinguishes SIMONE from the Literature

The following design choices collectively differentiate SIMONE from every comparable system reviewed in the literature:

1. **Package-level input** — SIMONE processes heterogeneous ZIP data packages (mixed file types), not scientific articles. No reviewed system handles this input type.
2. **Evidence-first extraction** — SIMONE separates evidence gathering (intermediate `ExtractionContext`) from profile construction. Other systems extract directly into the target schema.
3. **Staged vocabulary grounding** — Vocabulary normalization is a separate final stage after profile construction, using hybrid retrieval (vector + lexical + reciprocal-rank fusion + graph expansion). Most systems treat grounding as inline or post-processing.
4. **Domain-agnostic extraction with domain-specific grounding** — The extraction and projection stages contain no dataset-specific field names or vendor names. Domain specificity enters only through the configurable vocabulary layer.
5. **Provenance at chunk and source-text level** — Every extracted object carries a source-text snippet and file-level provenance, enabling traceability that most systems do not provide.
6. **Inspectable pipeline stages** — The staged design allows errors to be attributed to specific stages (chunking, extraction, merging, projection, grounding) rather than to an opaque LLM output.

---

## 3. Research Questions and How Each Chapter Answers Them

### 3.1 Central Research Question

> How can large language models be integrated into a controlled workflow for extracting, projecting, and semantically normalizing metadata from heterogeneous catalysis research data packages?

### 3.2 Supporting Sub-Questions and Chapter Mapping

| # | Sub-Question | Primary Answer in Chapter | Secondary Support |
|---|-------------|--------------------------|-------------------|
| SQ1 | Which metadata-relevant entities and attributes can be extracted from heterogeneous dataset packages using a chunk-wise LLM workflow? | Ch. 4 (Method) + Ch. 5 (Results) | Ch. 2 (theory: LLMs, structured output) |
| SQ2 | How can extraction outputs be made traceable to file-level and text-level evidence? | Ch. 4 (Method: provenance design) | Ch. 2 (theory: semantic modelling) |
| SQ3 | How can intermediate extraction objects be projected into an application profile such as ChemDCAT-AP? | Ch. 4 (Method: profile projection) | Ch. 3 (state of research: profiles, standards) |
| SQ4 | How can quantitative and qualitative attributes be normalized against controlled vocabularies (QUDT, Voc4Cat, CHMO, nmrCV)? | Ch. 4 (Method: vocabulary grounding) + Ch. 5 (Results) | Ch. 2 (theory: semantic web, querying) + Ch. 3 (state of research: vocabularies) |
| SQ5 | Which failure modes occur in extraction, profile construction, and final grounding, and how can they be evaluated? | Ch. 5 (Results: error analysis) | Ch. 3 (state of research: validation approaches) |

---

## 4. Chapter-by-Chapter Alignment

### Chapter 1: Introduction

**Central theme connection:** Establishes the problem (heterogeneous catalysis data lack machine-actionable metadata), the gap (manual extraction does not scale; unrestricted LLM use is insufficient), and the contribution (a controlled, staged workflow).

**Realistic goals:**
- Motivate why semantic metadata extraction from catalytic experiments matters for FAIR data reuse.
- State the central research question and supporting sub-questions clearly.
- Define scope: what SIMONE covers (ZIP packages, text-extractable files, configurable vocabularies) and what it excludes (images, full ontology engineering, production repository integration).
- Summarize the staged workflow approach in one paragraph.
- Provide a thesis outline that previews how each chapter contributes to answering the research questions.
- State the thesis contributions explicitly (3–5 bullet points).

**Consistency checkpoints:**
- The research questions stated here must match those referenced in Ch. 4 (method) and answered in Ch. 5 (results) and Ch. 6 (conclusion).
- The scope boundaries defined here must not be silently expanded in later chapters.
- The contributions listed here must be substantiated by evidence in Ch. 5.

**Current status:** Scaffold with sample paragraphs exists; needs polishing into a complete chapter.

---

### Chapter 2: Theoretical Background

**Central theme connection:** Provides the scientific and technical vocabulary needed to understand why a staged workflow is necessary and how each stage works. Every concept introduced here must be used in later chapters.

**Realistic goals:**
- 2.1 Catalytic Experiments: Explain the catalysis data landscape and why metadata is integral, not optional. Introduce NMR as a motivating example of why dataset records must preserve links between measurements, processing, instrument, sample, and experiment context.
- 2.2 Metadata Standards: Explain what a metadata standard is, why it provides the target structure for extraction, and why it enables validation.
- 2.3 Semantic Web Technologies: Introduce RDF, ontologies, SHACL, SPARQL, and graph querying as the representation and validation toolkit. Explain the schema-instance distinction and why URIs matter for grounding.
- 2.4 Large Language Models: Introduce transformer architecture, in-context learning, RAG, and structured-output methods. Explain why each technique is relevant to the SIMONE workflow.
- Avoid introducing concepts that are never referenced again. Every definition should have a downstream consumer.

**Consistency checkpoints:**
- The structured-output comparison table (prompting vs. tool calling vs. parsing/validation) must be referenced when Ch. 4 explains SIMONE's extraction and validation design.
- The RAG section must connect to Ch. 4's vocabulary retrieval strategy (vector + lexical + RRF + graph expansion).
- The SHACL section must connect to Ch. 4's validation stage.
- The ontology/RDF sections must connect to Ch. 3's discussion of ChemDCAT-AP, Voc4Cat, and other standards.
- NMR as a motivating example should reappear in Ch. 5 if NMR data packages are part of the evaluation set.

**Current status:** Revised through section 2.4. Sections are mature and well-structured.

---

### Chapter 3: Current State of Research

**Central theme connection:** Positions SIMONE within the existing landscape and identifies the research gap that the workflow addresses. This chapter must demonstrate that SIMONE's design choices are grounded in (or deliberately divergent from) the state of the art.

**Realistic goals:**
- 3.1 Standards and Resources for Semantic Metadata: Survey QUDT, Voc4Cat, CHMO, nmrCV, DCAT-AP+/ChemDCAT-AP, LinkML, SHACL, PROV-O, SKOS. Show how these standards provide the target structures that SIMONE populates.
- 3.2 Semantic Data Models for Catalysis: Survey NFDI4Cat data value chain, ChemDCAT-AP/LinkML, Ontologies4Cat, Reac4Cat, OntoCAPE, reporting standards. Show the infrastructure SIMONE targets.
- 3.3 NLP and LLM-Based Extraction Workflows: Progress from classical NLP (ChemDataExtractor) → domain-specific pre-training (MatSciBERT) → LLM extraction (CatMiner, SPIRES, Dagdelen, EDC, Schilling tutorial review). Compare architectural patterns. Position SIMONE's staged approach.
- 3.4 Validation and Human-in-the-Loop Approaches: Survey validation levels (syntactic, semantic, evidential, expert), provenance, confidence, human review. Connect to SIMONE's staged validation and curation stage.
- 3.5 Use Cases for Extracted Metadata (NEW SECTION): Show what happens downstream — KG population, FAIR repository deposit, AI-guided optimization, automated lab integration. This demonstrates that the value of extraction is measured by what it enables, not by extraction accuracy alone.
- 3.6 Research Gap: Explicitly state what no existing system does and how SIMONE fills that gap. Reference the six differentiators from §2.4 of this alignment document.

**Consistency checkpoints:**
- Every standard, vocabulary, or tool discussed here must either be used in Ch. 4 (as part of SIMONE's implementation) or serve as a comparison baseline.
- The research gap statement must directly motivate the design decisions described in Ch. 4.
- The comparison table of SIMONE vs. literature systems (from the chapter 3 guiding document) should be included or summarized.
- The "use cases for extracted metadata" section must connect SIMONE's output to the FAIR data infrastructure vision described in Ch. 1 and Ch. 2.

**Current status:** Empty placeholder. The chapter 3 guiding document provides a complete section-by-section plan with citation recommendations.

---

### Chapter 4: Methodical Approach

**Central theme connection:** This is the core design chapter. It must explain how each stage of the SIMONE workflow addresses a specific aspect of the central research question, and how each design decision is grounded in the literature reviewed in Ch. 3.

**Realistic goals:**
- 4.1 Workflow Requirements: Translate the research gap from Ch. 3 into functional and non-functional requirements. Define quality criteria (traceability, inspectability, semantic grounding, schema validation, human reviewability).
- 4.2 Overall Workflow Architecture: Present the staged pipeline with a clear diagram. Explain why a staged approach is chosen over a monolithic prompt (cite Ch. 3 evidence).
- 4.3 Data Package Processing: Describe upload, unpacking, text extraction, file ranking, and semantic chunking. Explain the design rationale for each step.
- 4.4 Extraction Context Design: Describe the intermediate `ExtractionContext` representation (DataGeneratingActivity, Method, EvaluatedEntity, AgenticEntity, Resource, QuantitativeAttribute, QualitativeAttribute). Explain why evidence is extracted into an intermediate representation rather than directly into the target profile.
- 4.5 Profile Projection and Semantic Reconstruction: Describe how merged evidence is projected into the ChemDCAT-AP-style profile. Explain coverage patching and reconstruction.
- 4.6 Vocabulary-Backed Normalization: Describe the hybrid retrieval strategy (vector + lexical + RRF + graph expansion), Neo4j-backed semantic service, and LLM-assisted candidate selection. Explain the separation between profile construction and vocabulary grounding.
- 4.7 Validation and Traceability: Describe schema validation, provenance preservation, and the staged error attribution model.
- 4.8 Human Review Interface: Describe the frontend inspection capabilities and the curation stage between profile draft and final grounding.

**Consistency checkpoints:**
- Every design decision must be traceable to a requirement in §4.1 and/or a literature reference from Ch. 3.
- The intermediate representation (ExtractionContext) must be consistent with the semantic modelling concepts from Ch. 2.3.
- The vocabulary grounding strategy must reference the RAG concepts from Ch. 2.4 and the retrieval strategy document.
- The validation approach must reference the structured-output comparison from Ch. 2.4.
- The profile projection must reference the metadata standards from Ch. 2.2 and the ChemDCAT-AP/LinkML discussion from Ch. 3.
- Keep implementation details (frameworks, versions, code structure) concise; move non-essential details to the appendix.

**Current status:** Empty placeholder. The working document and retrieval strategy document provide substantial material.

---

### Chapter 5: Results and Evaluation

**Central theme connection:** Provides the evidence that the staged workflow works, identifies where it fails, and measures how well each stage performs. This chapter substantiates (or challenges) the claims made in Ch. 1 and the design choices in Ch. 4.

**Realistic goals:**
- 5.1 Evaluation Setup: Describe the evaluation data packages, reference annotations, and assessment procedure.
- 5.2 Metrics: Report per-stage metrics — file-ranking quality, chunking usefulness, object extraction P/R/F1, attribute extraction P/R/F1, unit normalization success, quantity-kind normalization success, qualitative vocabulary grounding success, unresolved-but-correctly-unresolved rate, schema validation success, token usage, runtime.
- 5.3 Quantitative Results: Present results in tables organized by pipeline stage. Include baseline comparison (single-prompt extraction or manual extraction).
- 5.4 Qualitative Error Analysis: Categorize failures using the error taxonomy (missed files, bad chunks, over/under-extraction, wrong object types, wrong units, forced vocabulary mappings, schema-valid-but-wrong output, hallucinations).
- 5.5 Stage-Level Error Attribution: Show which stages contribute most to final output quality and which introduce the most errors. This is the key advantage of the staged design.
- 5.6 Comparison with Baseline: Compare SIMONE against at least one baseline (single-prompt extraction or manual extraction). If time is limited, compare against manual extraction only.

**Consistency checkpoints:**
- Every research sub-question from §3.2 must be addressed by at least one results table or analysis.
- The error categories must map back to the design decisions in Ch. 4 (e.g., "bad chunk boundary" tests the semantic chunking design; "forced vocabulary mapping" tests the grounding design).
- The metrics must be defined before they are reported — no undefined abbreviations.
- The evaluation set must respect the scope defined in Ch. 1 (heterogeneous data packages, not articles).
- If NMR data packages are used, connect back to the NMR example from Ch. 2.1.

**Current status:** Empty placeholder. The evaluation plan in the working document provides the framework.

---

### Chapter 6: Conclusion and Outlook

**Central theme connection:** Returns to the central research question, summarizes what was demonstrated, acknowledges limitations, and projects future development. This chapter must close the loop opened in Ch. 1.

**Realistic goals:**
- 6.1 Summary of Contributions: Restate the 3–5 contributions from Ch. 1 and cite the evidence from Ch. 5 that substantiates each.
- 6.2 Answers to Research Questions: Answer SQ1–SQ5 explicitly, referencing specific results from Ch. 5.
- 6.3 Limitations: Acknowledge what SIMONE does not do — image/binary extraction, full human-in-the-loop patch review, production repository integration, exhaustive ontology coverage. Connect limitations to the scope boundaries from Ch. 1.
- 6.4 Future Work: Project realistic next steps — complete correction workflow, file-format-specific parsers, OCR/multimodal, expanded vocabulary coverage, active-learning loops, batch extraction, RDF export, Repo4Cat integration, SHACL validation, model comparison, improved reproducibility tracking.
- 6.5 Broader Implications: Connect SIMONE to the bigger picture — FAIR catalysis data infrastructure, self-updating scientific knowledge graphs, AI-guided catalyst discovery. Show that the staged workflow pattern is transferable beyond catalysis.

**Consistency checkpoints:**
- The contributions summary must match the contributions stated in Ch. 1.
- The research question answers must reference specific Ch. 5 results, not vague claims.
- The limitations must not contradict the scope statement in Ch. 1 (i.e., do not apologize for things that were explicitly out of scope).
- The future work must be realistic for a post-thesis development horizon, not a grant proposal.

**Current status:** Empty placeholder.

---

## 5. Narrative Through-Lines

These are the threads that must remain visible in every chapter. If a paragraph does not advance at least one of these threads, it likely does not belong in the thesis.

### Through-Line 1: From Documents to Data

> Catalysis data are locked in heterogeneous, human-readable documents. The thesis develops a workflow that turns these documents into structured, machine-actionable data.

| Chapter | How this thread appears |
|---------|------------------------|
| Ch. 1 | Motivation: data packages are not machine-actionable |
| Ch. 2 | Theory: what machine-actionable metadata looks like (RDF, profiles, vocabularies) |
| Ch. 3 | State of the art: how others extract structure from documents |
| Ch. 4 | Method: SIMONE's pipeline for converting packages to structured metadata |
| Ch. 5 | Results: how well the conversion works |
| Ch. 6 | Conclusion: what was achieved and what remains |

### Through-Line 2: From Extraction to Semantics

> Extracted text is not yet semantic metadata. A field labelled "temperature" or "NMR" is ambiguous unless it is linked to a stable concept or unit identifier. The thesis separates extraction from semantic grounding.

| Chapter | How this thread appears |
|---------|------------------------|
| Ch. 1 | Gap: extraction alone is insufficient; grounding is needed |
| Ch. 2 | Theory: ontologies, vocabularies, URIs, SHACL |
| Ch. 3 | State of the art: standards and vocabularies that define the target semantics |
| Ch. 4 | Method: staged grounding (extraction → projection → vocabulary normalization) |
| Ch. 5 | Results: grounding success rates and unresolved mappings |
| Ch. 6 | Conclusion: semantic grounding as a key contribution |

### Through-Line 3: From Automation to Trust

> Fully automated extraction is the goal, but trust requires inspectability, provenance, and validation. The thesis designs a workflow that is automated where possible and inspectable where necessary.

| Chapter | How this thread appears |
|---------|------------------------|
| Ch. 1 | Positioning: SIMONE is an assistance system, not an autonomous replacement |
| Ch. 2 | Theory: validation mechanisms (SHACL, structured output, parsing) |
| Ch. 3 | State of the art: validation and human-in-the-loop approaches |
| Ch. 4 | Method: staged validation, provenance, frontend inspection, curation stage |
| Ch. 5 | Results: error analysis showing where trust is gained or lost |
| Ch. 6 | Conclusion: trust as a design principle; future work on complete review workflow |

### Through-Line 4: From Metadata to Discovery

> The value of extracted metadata is measured by what it enables downstream — FAIR repository deposit, knowledge graph population, AI-guided optimization — not by extraction accuracy alone.

| Chapter | How this thread appears |
|---------|------------------------|
| Ch. 1 | Motivation: FAIR data enable reuse and discovery |
| Ch. 2 | Theory: catalysis data landscape and the experimental chain |
| Ch. 3 | State of the art: use cases for extracted metadata (NEW section) |
| Ch. 4 | Method: output designed for repository deposit and KG ingestion |
| Ch. 5 | Results: schema-valid output that could be deposited |
| Ch. 6 | Conclusion: broader implications for catalysis data infrastructure |

---

## 6. Cross-Chapter Consistency Rules

### 6.1 Terminology

Use consistent terminology throughout the thesis. The following terms have specific meanings and should not be used interchangeably:

| Term | Definition | First Introduced |
|------|-----------|-----------------|
| **Data package** | A ZIP archive containing heterogeneous files from a catalysis experiment | Ch. 1 |
| **ExtractionContext** | The intermediate representation produced by chunk-wise LLM extraction | Ch. 4 |
| **Evidence candidate** | An extracted object paired with a source-text snippet | Ch. 4 |
| **Profile projection** | The process of mapping merged evidence into an application profile | Ch. 4 |
| **Vocabulary grounding** | The final stage of linking profile fields to controlled vocabulary URIs | Ch. 4 |
| **Semantic normalization** | Synonym for vocabulary grounding; use "vocabulary grounding" as the primary term | Ch. 4 |
| **Application profile** | A schema defining the target metadata structure (e.g., ChemDCAT-AP-style) | Ch. 2/3 |
| **Staged workflow** | SIMONE's pipeline design with separate, inspectable stages | Ch. 1/4 |
| **Inspectability** | The ability to examine intermediate results at each pipeline stage | Ch. 1/4 |
| **Traceability** | The ability to trace an extracted value back to its source text and file | Ch. 1/4 |

### 6.2 Scope Boundaries

The following scope boundaries are defined in Ch. 1 and must be respected throughout:

| In Scope | Out of Scope |
|----------|-------------|
| ZIP-based data packages with text-extractable files | Image-only or binary-only data packages |
| Text extraction from supported formats | OCR or multimodal extraction |
| Configurable vocabulary grounding (Voc4Cat, CHMO, nmrCV, QUDT) | Exhaustive ontology engineering for catalysis |
| Profile projection into ChemDCAT-AP-style schema | Production-grade repository integration |
| Frontend inspection and curation stage | Complete human-in-the-loop patch-review workflow |
| Open-source models via Ollama | Proprietary API-based models |
| Catalysis domain (with domain-agnostic extraction design) | Other experimental domains (discussed only as future work) |

### 6.3 Figure and Table Policy

- Figures that illustrate workflow architecture, pipeline stages, or data flow must be consistent across chapters. If a high-level pipeline diagram appears in Ch. 4, it should not be contradicted by a different representation in Ch. 5.
- Tables that compare SIMONE to literature systems (Ch. 3) must use the same terminology as the method descriptions in Ch. 4.
- Results tables in Ch. 5 must use metric names defined in Ch. 4 or Ch. 5.1, not ad hoc labels.

### 6.4 Citation Discipline

- Every citation in Ch. 2 (theory) should have a downstream consumer in Ch. 3, Ch. 4, or Ch. 5.
- Every design decision in Ch. 4 should have at least one literature justification from Ch. 3.
- Avoid introducing new citations in Ch. 5 or Ch. 6 that have not been discussed in earlier chapters.
- The bibliography should reflect the full literature corpus, but the thesis text should cite only what is directly relevant to the argument.

---

## 7. Realistic Goal-Setting Framework

### 7.1 Chapter Priority and Effort Allocation

Based on the working document's assessment and the thesis structure, the following priority ranking helps allocate remaining writing time:

| Priority | Chapter | Rationale | Estimated Effort |
|----------|---------|-----------|-----------------|
| 1 (Highest) | Ch. 4 — Methodical Approach | Core scientific contribution; prototype exists and is well-documented | High |
| 1 (Highest) | Ch. 5 — Results and Evaluation | Evidence-driven; requires running experiments and collecting metrics | High |
| 2 | Ch. 3 — Current State of Research | Positions the work; guiding document provides detailed plan | Medium-High |
| 2 | Ch. 1 — Introduction | Needs polishing from scaffold to complete chapter | Medium |
| 3 | Ch. 6 — Conclusion and Outlook | Depends on Ch. 5 results; can be drafted after Ch. 5 | Medium |
| 3 | Ch. 2 — Theoretical Background | Already revised through §2.4; may need minor adjustments for consistency | Low |

### 7.2 Minimum Viable Thesis

If time becomes constrained, preserve this core:

1. **Introduction** — clear research questions, scope, contributions
2. **Theoretical Background** — already largely complete
3. **Current State of Research** — standards, LLM extraction landscape, research gap
4. **Methodical Approach** — staged workflow description with architecture diagram
5. **Results and Evaluation** — at least one evaluation dataset, per-stage metrics, error analysis, one baseline comparison
6. **Conclusion and Outlook** — answers to research questions, limitations, future work

Implementation details that are not essential to the argument can be moved to the appendix.

### 7.3 Definition of Done per Chapter

A chapter is "done" when:

- [ ] It advances at least one narrative through-line (§5)
- [ ] Every section has a clear purpose stated in its opening paragraph
- [ ] Every technical term is defined on first use and used consistently (§6.1)
- [ ] Every citation is relevant and has a downstream consumer (§6.4)
- [ ] Scope boundaries are respected (§6.2)
- [ ] Cross-references to other chapters are correct and bidirectional
- [ ] Figures and tables are consistent with those in other chapters (§6.3)
- [ ] The chapter could be read by a reader who has read the preceding chapters but not the following ones

---

## 8. Bigger-Picture Positioning

### 8.1 Where SIMONE Sits in the Research Landscape

```
                    FAIR Data Infrastructure
                    (NFDI4Cat, Repo4Cat, NOMAD)
                           │
                           │  ← SIMONE output deposits here
                           │
    ┌──────────────────────┼──────────────────────┐
    │                      │                      │
    │   Semantic Standards │   Downstream AI      │
    │   (ChemDCAT-AP,      │   (BO-ICL, GOLLuM,   │
    │    Voc4Cat, QUDT)    │    CatTestHub, KGs)  │
    │                      │                      │
    │  ← SIMONE targets    │  ← SIMONE enables    │
    │    these standards   │    these use cases   │
    │                      │                      │
    └──────────────────────┼──────────────────────┘
                           │
                           │  ← SIMONE processes
                           │    these inputs
                           │
                    Heterogeneous Data Packages
                    (ZIPs, reports, tables, exports)
```

### 8.2 The Gap SIMONE Fills

The literature shows extensive work on:
- **Standards and infrastructure** (what the target should look like) — NFDI4Cat, ChemDCAT-AP, Voc4Cat
- **LLM-based extraction from articles** (how to get structure from papers) — CatMiner, SPIRES, Dagdelen
- **AI-guided discovery** (what to do with structured data) — BO-ICL, GOLLuM, ChemReasoner

What is missing is the bridge between **heterogeneous experimental data packages** and **standards-compliant metadata**. Existing extraction systems target published articles, not raw experimental data archives. Existing standards define what metadata should look like, but not how to produce it from unstructured packages at scale. SIMONE occupies this gap:

```
    Data Packages ──→ [ SIMONE ] ──→ Standards-Compliant Metadata ──→ FAIR Repositories / KGs / AI Discovery
    (unstructured)      (thesis)      (structured, grounded)             (downstream value)
```

### 8.3 Contribution to the Community

SIMONE contributes to the catalysis data community by:

1. **Demonstrating a viable path from raw data packages to FAIR metadata** — the path that NFDI4Cat infrastructure assumes but does not fully automate.
2. **Providing an inspectable, staged workflow pattern** — other domains can adapt the evidence-first extraction and separate grounding approach.
3. **Producing evidence-grounded, vocabulary-linked metadata** — not just structured JSON, but semantically interoperable records with traceable provenance.
4. **Identifying and categorizing failure modes** — the error taxonomy informs future work on automated metadata extraction for catalysis.

---

## 9. Quick-Reference Checklist for Chapter Writing

Before writing or revising any chapter, answer these questions:

1. **Which research sub-question(s) does this chapter address?** (See §3.2)
2. **Which narrative through-line(s) does this chapter advance?** (See §5)
3. **What is the reader expected to know after reading this chapter that they did not know before?**
4. **Which concepts from previous chapters does this chapter build on?**
5. **Which concepts from this chapter will later chapters reference?**
6. **Are all technical terms defined and used consistently?** (See §6.1)
7. **Are all scope boundaries respected?** (See §6.2)
8. **Does every section have a clear purpose that connects to the central theme?** (See §2)

---

## 10. Document Maintenance

This alignment document should be updated when:
- Research questions are finalized or revised
- Scope boundaries change
- The evaluation plan is concretized
- New literature is added that changes the positioning
- A chapter is completed and cross-references need verification

The document is a living reference, not a static artifact. Keep it current.