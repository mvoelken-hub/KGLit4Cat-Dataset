# Guiding Document: Chapter 3 — Current State of Research

> This document maps the existing chapter 3 against the full literature corpus (LLM Wiki) and the SIMONE prototype, identifies gaps, and provides concrete guidance for revision.

---

## 1. Chapter 3 — Current Structure and Assessment

### Existing sections

| # | Section | Status | Quality |
|---|---------|--------|---------|
| 3.1 | Standards and Resources for Semantic Metadata | Complete | Strong — covers QUDT, Voc4Cat, CHMO, nmrCV, DCAT-AP+/ChemDCAT-AP, LinkML, SHACL, PROV-O, SKOS |
| 3.2 | Semantic Data Models for Catalysis | Complete | Strong — covers NFDI4Cat, Repo4Cat, ChemDCAT-AP, Voc4Cat, Metadata4Cat, Ontologies4Cat, Reac4Cat, OntoCAPE, reporting standards |
| 3.3 | NLP and LLM-based Extraction Workflows | Complete | Good but needs expansion — covers CatalysisIE, CataLM, CatMiner, SPIRES, Dagdelen, RAG, but misses several newly added sources |
| 3.4 | Human-in-the-loop and Validation Approaches | Complete | Adequate — covers validation levels, provenance, confidence, but could be more concrete |
| 3.5 | Research Gap | Complete | Good — clearly positions the thesis, but could reference more of the new literature |

### What's missing

1. **Real use cases for extracted metadata** — The chapter describes extraction methods but does not show what happens downstream: how extracted metadata populates knowledge graphs, enables catalyst discovery, feeds FAIR repositories, or supports AI-guided optimization. This is a critical gap the user identified.

2. **Newly added literature not yet cited** — 16 new PDFs were added to the corpus. Several are directly relevant to chapter 3:
   - `schilling2025text` (From Text to Insight) — comprehensive tutorial review of LLM-based chemical data extraction
   - `zhang2024chemicaltextmining` (Fine-tuning LLMs for chemical text mining) — comparison of fine-tuning approaches
   - `dagdelen2024structured` (already cited but could be expanded)
   - `swain2016chemdataextractor` (ChemDataExtractor) — foundational pre-LLM tool
   - `gupta2022matscibert` (MatSciBERT) — domain-specific pre-training
   - `zhang2024edc` (EDC) — LLM-based KG construction
   - `marshall2023digital` (Achieving Digital Catalysis) — vision for digital catalysis data
   - `mendes2021opendata` (Open Data in Catalysis) — big data vs small data
   - `moshantaf2024advancing` (FAIR data in local infrastructure) — practical FAIR implementation
   - `cattesthub` (CatTestHub) — benchmarking database
   - `ding2026scikg` (KG survey for AI for science) — already cited but could be expanded

3. **Explicit SIMONE comparison** — The chapter positions the research gap well but does not explicitly compare SIMONE's design choices against each major approach in the literature.

---

## 2. SIMONE vs. Literature — Detailed Comparison

### 2.1 Metadata Standards Landscape

| Dimension | SIMONE | Literature State of the Art | Key References |
|-----------|--------|------------------------------|----------------|
| **Target profile** | DCAT-AP+ / ChemDCAT-AP-style profile, schema-validated via JSON Schema | ChemDCAT-AP is the emerging standard; LinkML compiles to SHACL, JSON-LD, Pydantic | `stroemert2026chemdcat`, `chemdcatapDocumentation` |
| **Quantitative grounding** | QUDT quantity kinds and units; deterministic class assignment | QUDT is standard; ChemDCAT-AP references it for attribute patterns | `qudt2022schema`, `stroemert2026chemdcat` |
| **Qualitative grounding** | Configurable vocabularies (Voc4Cat, CHMO, nmrCV); vector + lexical retrieval with reciprocal-rank fusion | Voc4Cat is community standard; CHMO for methods; nmrCV for NMR | `nfdi4cat2024whitepaper`, `chmoOboFoundry`, `nmrmlSpecification` |
| **Provenance** | Evidence candidates carry `evidence_text` and `source_context`; file-level provenance preserved | PROV-O is the standard model; ChemDCAT-AP uses it for data-generating activities | `w3c2013provo`, `stroemert2026chemdcat` |
| **Validation** | JSON Schema validation + SHACL-ready (via LinkML compilation) | LinkML → SHACL is the recommended path; validation is standard practice | `w3c2017shacl`, `stroemert2026chemdcat` |
| **Repository integration** | Prototype-level export; not yet integrated into Repo4Cat or NOMAD | Repo4Cat (NFDI4Cat repository), NOMAD catalysis plugin, Chemotion ELN (NFDI4Chem) | `kushnarenko2025repo4cat`, `schumann2025nomad`, `steinbeck2023nfdi4chem` |

**SIMONE's position:** SIMONE operates *between* the heterogeneous source material and the semantic standards. It does not propose new standards — it uses existing ones (ChemDCAT-AP, QUDT, Voc4Cat) as target constraints. This is a deliberate design choice that differentiates SIMONE from ontology-development work (e.g., `behr2024ontologies4cat`, `borgelt2025ontology`) and aligns it with application-profile-driven extraction.

### 2.2 LLM-Based Extraction Approaches

| Dimension | SIMONE | CatMiner | CatalysisIE | SPIRES | Dagdelen et al. | EDC |
|-----------|--------|----------|-------------|--------|-----------------|-----|
| **Input** | Heterogeneous ZIP packages (PDFs, tables, text, images) | Scientific articles (PDF text) | Full-text journal articles | Text documents | Scientific sentences/paragraphs | Text documents |
| **Architecture** | Staged: ingest → text extraction → chunking → evidence extraction → profile projection → grounding → validation | Multi-turn QA: iterative prompting with domain knowledge, chat memory, paragraph search | Classical NLP pipeline: article selection → NER → relation extraction | Recursive LLM prompt interrogation against LinkML schema | Fine-tuned GPT-3/Llama-2 for joint NER + RE | Three-phase: open IE → schema definition → canonicalization |
| **Output** | Schema-validated metadata profile with vocabulary-grounded terms | Structured CSV records (catalyst, conditions, performance) | Entity annotations + entity-aware search engine | Populated LinkML schema instances | JSON/English structured records | Knowledge graph triplets |
| **Provenance** | Source-text evidence per extracted candidate; file-level provenance | Not explicitly tracked | Entity-level annotations linked to source | Grounded to ontology IDs | Not explicitly tracked | Triple-level provenance |
| **Vocabulary grounding** | Separate final stage: vector + lexical retrieval, reciprocal-rank fusion, Neo4j graph expansion, LLM candidate selection | Post-processing normalization step | Not part of the pipeline | Uses ontologies for entity grounding via OntoPortal | Not part of the pipeline | Post-hoc canonicalization against schema |
| **Validation** | JSON Schema + SHACL-ready; staged (syntactic, semantic, evidential, expert) | Reporting standards suggested | Evaluation against annotated corpus | Schema-conformant output | Not explicitly validated | Schema-conformant output |
| **Domain specificity** | Domain-agnostic by design; configurable vocabularies | Catalysis-specific (OCM case study) | Catalysis-specific (6 entity types) | Domain-agnostic (any LinkML schema) | Materials chemistry | Domain-agnostic |
| **Chunking** | Semantic embedding-distance breakpoints + fixed-token fallback | Paragraph-level | Document-level | Full document | Sentence/paragraph | Full document |
| **Human-in-the-loop** | Curation stage between profile draft and grounding; frontend inspection | Suggested community reporting standards | Active learning for uncertain examples | Expert review of extracted records | Not part of the system | Not part of the system |
| **Model choice** | Open-source (Ollama); domain-agnostic small model | LLM-agnostic (GPT, Llama, DeepSeek) | Task-specific trained models | GPT-3.5/GPT-4 | GPT-3, Llama-2 (fine-tuned) | Any LLM (zero-shot) |

**Key differentiators of SIMONE:**
1. **Package-level, not article-level** — SIMONE processes heterogeneous dataset archives (ZIPs with mixed file types), not just scientific articles. No other system in the literature handles this input type.
2. **Evidence-first extraction** — SIMONE separates evidence gathering from profile construction. Other systems (CatMiner, Dagdelen, SPIRES) extract directly into the target schema.
3. **Staged vocabulary grounding** — SIMONE performs vocabulary grounding as a separate final stage after profile construction, using hybrid retrieval (vector + lexical + graph expansion). CatMiner treats normalization as post-processing; SPIRES grounds during extraction.
4. **Domain-agnostic design** — SIMONE's extraction and projection filters are explicitly domain-agnostic (no dataset-specific field names, vendor names, or instrument names). This contrasts with CatalysisIE (catalysis-specific entity types) and CatMiner (catalysis-specific schema).
5. **Measurement evidence routing** — SIMONE routes measurement evidence through a structured semantic-routing call that determines activity/entity attribute placement. No other system handles this parent-ownership decision.

### 2.3 Fine-Tuning vs. Prompting vs. Agent Approaches

| Approach | Representative | How SIMONE relates |
|----------|---------------|-------------------|
| **Fine-tuning** | `zhang2024chemicaltextmining` (GPT-3.5, Mistral, Llama3 fine-tuned on 5 chemical tasks), `dagdelen2024structured` (GPT-3, Llama-2 fine-tuned for materials RE) | SIMONE uses zero-shot/few-shot prompting without fine-tuning. This is a tradeoff: lower setup cost but potentially lower accuracy on domain-specific tasks. Fine-tuned models achieve 69–95% accuracy; SIMONE's accuracy is evaluation-dependent. |
| **In-context learning** | `ramos2025boicl` (BO-ICL: frozen LLMs as regression models), `rankovic2025uncertainty` (GOLLuM: LLM embeddings → GP) | SIMONE uses in-context learning but with staged prompting (evidence extraction ≠ profile projection ≠ grounding). BO-ICL/GOLLuM focus on optimization, not extraction. |
| **Agentic / structured search** | `sprueill2023montecarlo` (ChemReasoner: MCTS over prompts), `ock2025adsorbagent` (Adsorb-Agent: LLM agent for adsorption search) | SIMONE is not agentic — it follows a fixed pipeline. Agentic approaches allow adaptive exploration but sacrifice reproducibility and traceability. SIMONE prioritizes inspectability over adaptivity. |
| **Tutorial review perspective** | `schilling2025text` (From Text to Insight) | This review provides the framework for understanding SIMONE's design: end-to-end workflow (data collection → structured output), quality assurance (constrained decoding, domain validation), and future frontiers (agentic systems, multimodal). |

---

## 3. Guidance for Chapter Revision

### 3.1 Section-by-Section Recommendations

#### Section 3.1: Standards and Resources for Semantic Metadata
**Status:** Strong. Minor updates needed.

**Add:**
- Brief mention of `marshall2023digital` as the vision for "digital catalysis" that motivates these standards.
- Reference to `moshantaf2024advancing` as a practical example of local FAIR data infrastructure implementation (EPICS, automated reactor, SOPs).
- Reference to `huskova2025improvement` for the use-case-driven methodology for metadata quality in NFDI4Cat.
- Note that `steinbeck2020nfdi4chem` (NFDI4Chem grant proposal) outlines the six key objectives that led to the infrastructure SIMONE targets.

**Keep as-is:** QUDT, Voc4Cat, CHMO, nmrCV, DCAT-AP+/ChemDCAT-AP descriptions are well-written.

#### Section 3.2: Semantic Data Models for Catalysis
**Status:** Strong. Could be slightly expanded.

**Add:**
- `mendes2021opendata` — the "big data vs small data" framing. This is relevant because SIMONE addresses "small data" scenarios (few dataset packages, not millions of papers).
- `cattesthub` — CatTestHub as an example of benchmarking databases that benefit from structured metadata. 250+ data points, 24 catalysts, 3 probe chemistries. Shows what structured catalysis data looks like in practice.
- `schumann2025nomad` (NOMAD catalysis plugin) — already cited but could be expanded as an example of a repository-level tool that consumes structured metadata.

#### Section 3.3: NLP and LLM-based Extraction Workflows
**Status:** Good but needs expansion with new literature.

**Add a subsection on pre-LLM and domain-specific NLP tools:**
- `swain2016chemdataextractor` (ChemDataExtractor) — foundational pre-LLM toolkit. Chemistry-aware NLP pipeline with rule-based grammars, document-level processing. F-scores 87–93%. This is the baseline that LLM approaches improve upon.
- `gupta2022matscibert` (MatSciBERT) — BERT pre-trained on materials science corpus. Outperforms SciBERT. Shows that domain-specific pre-training matters. Relevant to SIMONE because it motivates domain adaptation but SIMONE uses domain-agnostic prompting.

**Add a subsection on fine-tuning approaches:**
- `zhang2024chemicaltextmining` — fine-tuned GPT-3.5/Mistral/Llama3 on five chemical text mining tasks. 69–95% accuracy with minimal annotated data. Key comparison: fine-tuning outperforms prompt engineering, but SIMONE avoids fine-tuning for transferability.
- `dagdelen2024structured` — already cited; expand to note the joint NER+RE approach and structured JSON output.

**Add a subsection on LLM-based KG construction:**
- `zhang2024edc` (EDC) — three-phase framework for KG construction. Relevant to SIMONE because EDC also separates extraction from canonicalization, but EDC targets triplets while SIMONE targets profile attributes.
- `schilling2025text` (From Text to Insight) — comprehensive tutorial review. Use as the framing reference for the LLM extraction landscape: end-to-end workflows, quality assurance, constrained decoding, domain-specific validation.

**Add a paragraph on the fine-tuning vs. prompting tradeoff:**
- `llama3-lora-qlora` provides technical background on LoRA/QLoRA. `zhang2024chemicaltextmining` shows that fine-tuned open-source models (Mistral, Llama3) are competitive with GPT-3.5. SIMONE's choice to use zero-shot prompting is a deliberate tradeoff: lower setup cost and higher transferability, but potentially lower accuracy on domain-specific tasks.

**Add a paragraph on the extraction-to-KG pipeline:**
- `ding2026scikg` (KG survey) — already cited; expand to describe the vision of self-updating SciKGs that LLMs feed.
- `oarga2026scientifickg` — already cited; highlight that zero-shot LLM KG+ontology generation is possible and that SIMONE's evidence-grounded approach addresses the hallucination concern raised there.

#### Section 3.4: Human-in-the-loop and Validation Approaches
**Status:** Adequate. Could be more concrete.

**Add:**
- `schilling2025text` — the tutorial review's quality assurance framework: constrained decoding and domain-specific validation. This directly maps to SIMONE's staged validation.
- `marconato2025reasoning` — reasoning shortcuts in neuro-symbolic models. Relevant to the validation discussion: even schema-valid output can exploit shortcuts rather than representing genuine understanding. This is a theoretical justification for SIMONE's evidence-grounding requirement.
- `kim2025materialsdiscovery` (AlchemyBench) — LLM-as-a-Judge framework for automated evaluation. Relevant to the validation discussion as a potential evaluation method for SIMONE.

#### Section 3.5: New Section — Use Cases for Extracted Metadata
**Status:** MISSING. This is the critical gap.

This section should answer: "What happens after metadata is extracted and grounded?" It should show that the semantic infrastructure is not an end in itself but enables concrete downstream use cases.

**Recommended content:**

**3.5.1 Knowledge Graph Population for Catalyst Discovery**
- Extracted metadata (catalyst, conditions, performance) populates KGs that support catalyst recommendation and reaction mechanism discovery (`diaz2025knowledgegraphs`).
- KG + ontology improves RAG quality vs KG-only (`oarga2026scientifickg`).
- Vision: self-updating SciKGs co-evolving with LLMs (`ding2026scikg`).
- **SIMONE connection:** SIMONE's profile output (ChemDCAT-AP-style) is designed to be ingestible into such KGs. The staged grounding ensures that terms carry vocabulary URIs, not raw strings.

**3.5.2 FAIR Data Repository Population**
- Structured metadata enables dataset deposit in FAIR repositories (Repo4Cat, NOMAD, Chemotion).
- Repo4Cat supports working spaces for industry collaboration with IP protection (`kushnarenko2025repo4cat`).
- NOMAD catalysis plugin enables structured upload with Voc4Cat alignment (`schumann2025nomad`).
- NFDI4Chem Smart Lab concept: ELN → repository pipeline (`steinbeck2023nfdi4chem`).
- Practical example: Moshantaf et al. (`moshantaf2024advancing`) implemented automated data acquisition → analysis → database upload → relationship generation for a catalytic test reactor using EPICS.
- **SIMONE connection:** SIMONE's output is a schema-valid metadata document that could be deposited in such repositories. The curation stage allows human review before deposit.

**3.5.3 AI-Guided Catalyst Optimization**
- Extracted structure-environment-property data feeds Bayesian optimization for catalyst discovery.
- BO-ICL (`ramos2025boicl`) uses natural language representations of experiments — directly relevant if metadata is extracted in structured form first.
- GOLLuM (`rankovic2025uncertainty`) uses LLM embeddings of experimental descriptions for optimization — structured metadata could replace ad hoc text descriptions.
- CatTestHub (`cattesthub`) provides benchmarking data for comparing catalysts — structured extraction enables populating such databases from literature.
- **SIMONE connection:** SIMONE's quantitative attribute grounding (QUDT) ensures units and quantity kinds are normalized, which is essential for feeding optimization algorithms. Without normalization, values like "350°C" vs "623K" cannot be compared.

**3.5.4 Automated Laboratory Integration**
- FAIR metadata enables closed-loop experimentation: design → build → test → learn.
- Autonomous protein engineering (`weigmann2026autonomous`) demonstrates DBTL cycles reduced from months to days.
- Moshantaf et al. (`moshantaf2024advancing`) show automated reactor → database → API pipeline.
- Catalyst informatics (`takahashi2023catalysts`) envisions ontology-driven catalyst design platforms.
- **SIMONE connection:** SIMONE does not implement closed-loop experimentation, but its structured output is a prerequisite for such systems. The workflow produces metadata that machines can consume, not just documents humans can read.

#### Section 3.6: Research Gap (renumbered from 3.5)
**Status:** Good. Minor updates.

**Add:**
- Reference to `schilling2025text` — the tutorial review identifies the lack of standardized guidelines for LLM-based chemical data extraction as an open problem. SIMONE's staged workflow with validation is a response to this gap.
- Reference to `mendes2021opendata` — the "small data" problem. SIMONE addresses scenarios where few dataset packages exist, not large-scale literature mining.
- Strengthen the positioning: SIMONE is the only system that handles *heterogeneous dataset packages* (not articles) and separates *evidence extraction* from *profile construction* from *vocabulary grounding* as distinct stages with provenance.

---

## 4. SIMONE Design Decisions — Literature Justification Map

This table maps each key SIMONE design decision to the literature that justifies (or challenges) it. Use this when writing the chapter to show that design choices are grounded in the state of the art.

| SIMONE Design Decision | Literature Support | Literature Challenge | Implication for Chapter |
|------------------------|--------------------|-----------------------|------------------------|
| **Staged workflow (not monolithic prompt)** | CatMiner uses multi-turn QA (`walls2025catminer`); SPIRES uses recursive extraction (`caufield2024spires`); EDC uses three-phase (`zhang2024edc`) | Dagdelen et al. show single-pass extraction is possible with fine-tuning (`dagdelen2024structured`) | Cite staged approaches as a design pattern; acknowledge single-pass as a baseline alternative |
| **Evidence-first extraction** | CatMiner notes that paragraph-level context matters (`walls2025catminer`); provenance tracking recommended by `diaz2025knowledgegraphs` | No system explicitly separates evidence from profile construction like SIMONE does | Frame as SIMONE's contribution; cite provenance recommendations as motivation |
| **Semantic chunking** | Lost in the Middle problem (`liu2024lostmiddle`); context management is an explicit concern in `schilling2025text` | Fine-tuned models can handle longer context (`zhang2024chemicaltextmining`) | Cite context management as motivation; acknowledge that larger context windows may reduce need for chunking |
| **Domain-agnostic extraction** | SPIRES is domain-agnostic (`caufield2024spires`); EDC is domain-agnostic (`zhang2024edc`) | Domain-specific models outperform general ones: CataLM (`wang2024catalm`), MatSciBERT (`gupta2022matscibert`), fine-tuned LLMs (`zhang2024chemicaltextmining`) | Frame as a tradeoff: transferability vs. accuracy; cite domain-specific results as upper bound |
| **Separate vocabulary grounding stage** | CatMiner treats normalization as post-processing (`walls2025catminer`); SPIRES grounds during extraction (`caufield2024spires`) | No direct challenge, but grounding quality depends on vocabulary coverage (`behr2024ontologies4cat`) | Frame as a design choice that enables vocabulary-agnostic profile construction |
| **Schema validation (JSON Schema + SHACL-ready)** | LinkML → SHACL is recommended by ChemDCAT-AP (`stroemert2026chemdcat`); validation is standard practice | Schema validity ≠ scientific correctness (acknowledged in `diaz2025knowledgegraphs`, `rankovic2025uncertainty`) | Cite as necessary but insufficient; connect to human-in-the-loop section |
| **Human-in-the-loop (curation stage)** | Recommended by `diaz2025knowledgegraphs`, `walls2025catminer`, `rankovic2025uncertainty`; AlchemyBench uses LLM-as-a-Judge (`kim2025materialsdiscovery`) | Autonomous systems aim to reduce human involvement (`weigmann2026autonomous`, `moshantaf2024advancing`) | Frame as a spectrum: SIMONE reduces manual burden but does not eliminate expert review |
| **Package-level input (ZIPs)** | No comparable system in the literature | — | Frame as SIMONE's unique contribution; cite `marshall2023digital` and `moshantaf2024advancing` as motivation (local lab data are heterogeneous) |
| **Measurement evidence routing** | DCAT-AP+ activity/entity patterns (`stroemert2026chemdcat`); Reac4Cat reaction classification (`borgelt2025ontology`) | No system handles this routing decision automatically | Frame as a contribution; cite DCAT-AP+ activity patterns as the target structure |

---

## 5. Recommended Citation Additions

The following citations from the newly added literature should be integrated into chapter 3. They are grouped by the section where they should appear.

### Section 3.1 (Standards)
- `marshall2023digital` — vision for digital catalysis
- `moshantaf2024advancing` — practical FAIR data implementation
- `huskova2025improvement` — metadata quality methodology
- `steinbeck2020nfdi4chem` — NFDI4Chem objectives

### Section 3.2 (Semantic Data Models)
- `mendes2021opendata` — big data vs small data framing
- `cattesthub` — benchmarking database example
- `schumann2025nomad` — expand NOMAD reference

### Section 3.3 (NLP/LLM Extraction)
- `swain2016chemdataextractor` — foundational pre-LLM tool
- `gupta2022matscibert` — domain-specific pre-training
- `zhang2024chemicaltextmining` — fine-tuning comparison
- `schilling2025text` — comprehensive tutorial review
- `zhang2024edc` — LLM-based KG construction
- `lin2024knnbioel` — entity linking methodology
- `kim2025angel` — negative sample learning for entity linking
- `zi2026shattering` — LLM reasoning limitations with KG topology
- `marconato2025reasoning` — reasoning shortcuts in concept learning
- `llama3-lora-qlora` — LoRA/QLoRA technical background

### Section 3.4 (Validation)
- `schilling2025text` — quality assurance framework
- `marconato2025reasoning` — reasoning shortcuts
- `kim2025materialsdiscovery` — LLM-as-a-Judge

### Section 3.5 (New: Use Cases)
- `ding2026scikg` — KG vision for AI for science
- `oarga2026scientifickg` — KG + ontology + RAG
- `kushnarenko2025repo4cat` — repository infrastructure
- `schumann2025nomad` — NOMAD catalysis plugin
- `moshantaf2024advancing` — automated lab → database pipeline
- `ramos2025boicl` — BO with natural language
- `rankovic2025uncertainty` — LLM embeddings for optimization
- `cattesthub` — benchmarking database
- `weigmann2026autonomous` — autonomous experimentation
- `takahashi2023catalysts` — catalyst informatics vision

### Section 3.6 (Research Gap)
- `schilling2025text` — lack of standardized guidelines
- `mendes2021opendata` — small data problem
- `zhang2024edc` — extraction-canonicalization separation

---

## 6. Suggested Chapter Outline (Revised)

```
3. Current State of Research

3.1 Standards and Resources for Semantic Metadata
    3.1.1 Semantic Web Foundations (RDF, RDFS, OWL, SKOS, SHACL, PROV-O)
    3.1.2 Quantitative Vocabularies (QUDT)
    3.1.3 Catalysis Vocabularies (Voc4Cat, CHMO, nmrCV)
    3.1.4 Application Profiles (DCAT-AP+, ChemDCAT-AP, LinkML)
    3.1.5 FAIR Data Infrastructure (NFDI4Cat, NFDI4Chem, Repo4Cat, NOMAD)
    [Add: marshall2023digital vision, moshantaf2024advancing practical impl., 
     huskova2025improvement quality methodology]

3.2 Semantic Data Models for Catalysis
    3.2.1 NFDI4Cat Data Value Chain
    3.2.2 ChemDCAT-AP and LinkML
    3.2.3 Ontology Landscape (Ontologies4Cat, Reac4Cat, OntoCAPE)
    3.2.4 Reporting Standards and Minimum Information
    [Add: mendes2021opendata big/small data, cattesthub benchmarking DB]

3.3 NLP and LLM-Based Extraction Workflows
    3.3.1 Classical NLP Pipelines (ChemDataExtractor, CatalysisIE)
    3.3.2 Domain-Specific Pre-training (MatSciBERT, CataLM)
    3.3.3 LLM-Based Structured Extraction (Dagdelen, SPIRES, EDC)
    3.3.4 Domain-Specific LLM Extraction (CatMiner)
    3.3.5 Fine-Tuning vs. Prompting (Zhang et al., Schilling tutorial)
    3.3.6 Retrieval-Augmented Generation for Extraction
    [NEW: Add swain2016chemdataextractor, gupta2022matscibert, 
     zhang2024chemicaltextmining, schilling2025text, zhang2024edc,
     llama3-lora-qlora]

3.4 Validation and Human-in-the-Loop Approaches
    3.4.1 Validation Levels (syntactic, semantic, evidential, expert)
    3.4.2 Schema Validation (JSON Schema, SHACL, LinkML)
    3.4.3 Provenance and Confidence
    3.4.4 Human Review and Curation
    [Add: schilling2025text QA framework, marconato2025reasoning shortcuts,
     kim2025materialsdiscovery LLM-as-a-Judge]

3.5 Use Cases for Extracted Metadata  [NEW SECTION]
    3.5.1 Knowledge Graph Population for Catalyst Discovery
    3.5.2 FAIR Data Repository Population
    3.5.3 AI-Guided Catalyst Optimization
    3.5.4 Automated Laboratory Integration

3.6 Research Gap
    [Update with new references and SIMONE positioning]
```

---

## 7. Key Narrative Threads

When writing the chapter, maintain these narrative threads that connect the literature to SIMONE:

1. **From documents to data:** The literature shows a progression from manual curation → rule-based NLP (ChemDataExtractor) → domain-specific pre-training (MatSciBERT) → LLM-based extraction (CatMiner, Dagdelen, SPIRES). SIMONE extends this progression to *heterogeneous dataset packages*.

2. **From extraction to semantics:** Extraction alone is insufficient. The extracted data must be grounded in vocabularies (Voc4Cat, QUDT, CHMO), validated against schemas (ChemDCAT-AP, LinkML), and connected to provenance (PROV-O). SIMONE operationalizes this chain.

3. **From metadata to discovery:** Structured metadata enables downstream use cases: KG population, FAIR repository deposit, AI-guided optimization, and automated experimentation. The value of extraction is measured by what it enables downstream, not by extraction accuracy alone.

4. **From automation to trust:** Fully automated extraction is the goal but not yet the reality. The literature converges on staged workflows with provenance, confidence handling, and human review. SIMONE's design reflects this consensus.

5. **From big data to small data:** Large-scale literature mining (millions of papers) is one paradigm. But many catalysis research scenarios involve *small data* — a few dataset packages from a specific experiment. SIMONE addresses this scenario directly.