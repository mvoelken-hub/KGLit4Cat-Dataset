# LLM Wiki — Schema & Operating Instructions

> Based on the [LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) by Andrej Karpathy.

## Recent Updates

### [2026-07-01] ingest | ChemSpectra analytical-data tooling
Added source page for newly acquired PDF:
- [[chemspectra]] - web-based spectra editor for IR/MS/NMR, open JCAMP-DX/mzML formats, Chemotion ELN/repository integration, and FAIR analytical-data workflows

Updated concept/entity pages: catalysis-data-infrastructure, nfdi4chem. Updated overview, index, log, and thesis bibliography references.

### [2026-06-29] ingest | Chemistry IE and generative IE survey batch
Added source pages for newly acquired PDFs:
- [[openchemie]] - multimodal document-level reaction extraction from chemistry literature across text, tables, and figures
- [[llm-generative-information-extraction-survey]] - LLM-based generative IE survey covering NER, relation extraction, event extraction, universal IE, low-resource techniques, retrieval, and reliability issues

Marked [[zhang2024edc]] as already covered; `2404.03868v2.pdf` is a duplicate/preprint of the existing EDC source. Updated text-mining and structured-output concept pages, overview, index, log, and thesis bibliography references.

### [2026-06-28] ingest | Reac4Cat-Ontology
Added source page for newly acquired PDF:
- [[reac4cat-ontology]] - OWL description logic, General Class Axioms, reaction-role/catalyst inference, EnzymeML and DWSIM-linked process-simulation knowledge graph

Updated concept/entity pages: ontology-development, catalysis-ontologies. Updated overview, index, log, and thesis bibliography references.

### [2026-06-27] ingest | Biocatalysis and catalysis data infrastructure batch
Added source pages for newly acquired PDFs:
- [[enzymeml-fair-biocatalysis]] - EnzymeML exchange format, STRENDA-aligned metadata, APIs, ELNs, Dataverse micropublications
- [[fair-biocatalytic-science]] - FAIR data/software, exchange formats, ontologies, ELNs/LIMSs, repositories, FAIR software for biocatalysis
- [[electrocatalysis-research-database]] - Catalysis-Hub Experimental database with electrocatalysis metadata, spectra, curves, web/API access
- [[catalysis-data-infrastructure-ukch]] - UKCH Catalysis Data Infrastructure catalogue linking publications, datasets, authors, institutions, and themes

Updated concept page: catalysis-data-infrastructure. Updated overview, index, log, and thesis bibliography references.

### [2026-06-27] ingest | FAIR implementation and materials infrastructure batch
Added source pages for newly acquired PDFs:
- [[fair-cookbook]] - practical FAIRification recipes, maturity indicators, metadata/provenance/licensing guidance
- [[chemical-data-storage-architectures]] - chemical databases, schemas, ETL, APIs, and FAIR laboratory data
- [[fair-biopharma-rd]] - FAIR implementation in industrial R&D, governance, stewardship, knowledge representation
- [[fair-digital-twins]] - FAIR for digital twins, Semantic Web technologies, authentication/authorization, provenance
- [[fairmat-materials-research]] - FAIRmat/NOMAD materials infrastructure, federated repositories, metadata and ontologies

Updated concept page: catalysis-data-infrastructure. Updated overview, index, and log.

### [2026-06-26] ingest | LLM foundations, RAG, and structured-output batch
Added source pages for fresh PDFs:
- [[attention-is-all-you-need]] - Transformer architecture
- [[bm25-probabilistic-relevance]] - BM25 and probabilistic relevance
- [[gpt3-few-shot-learners]] - few-shot prompting / in-context learning
- [[retrieval-augmented-generation-knowledge-intensive-nlp]] - original RAG formulation
- [[sentencepiece]] - language-independent subword tokenization
- [[bert]], [[scibert]], [[sentence-bert]] - encoder and embedding foundations
- [[neural-text-degeneration]] - decoding failure modes and nucleus sampling
- [[instructgpt-rlhf]] - instruction following with human feedback
- [[toolformer]] - learned tool use
- [[pagedattention-vllm]] - LLM serving infrastructure
- [[grammar-constrained-decoding]], [[structured-output-ie-go]] - structured output reliability
- [[rag-survey]], [[self-rag]], [[lost-in-the-middle]] - RAG and long-context evidence use
- [[llm-kg-extraction-tables-materials]] - table-to-KG extraction in materials science
- [[llm-survey-zhao2026]] - broad LLM survey

Created concept pages: llm-foundations, retrieval-augmented-generation, structured-output-reliability
Updated concept pages: chemical-llms, text-mining-for-catalysis, knowledge-graphs-for-catalysis, automated-scientific-discovery

### [2025-06-23] ingest | 16 new sources added
Added source pages for newly acquired PDFs:
- [[cattesthub]] — CatTestHub benchmarking database
- [[huskova2025improvement]] — Data/metadata quality methodology (NFDI4Cat)
- [[zhang2024edc]] — EDC: LLM-based KG construction (EMNLP 2024)
- [[kim2025angel]] — ANGEL: Negative sample learning for BioEL
- [[zi2026shattering]] — ShatterMed-QA: Multi-hop medical reasoning benchmark
- [[marshall2023digital]] — Achieving Digital Catalysis (Angew. Chem.)
- [[mendes2021opendata]] — Open Data in Catalysis: Big Picture to Small Data
- [[swain2016chemdataextractor]] — ChemDataExtractor toolkit
- [[schilling2025text]] — From Text to Insight: LLMs for chemical data extraction
- [[moshantaf2024advancing]] — FAIR data in local catalysis infrastructure
- [[zhang2024chemicaltextmining]] — Fine-tuning LLMs for chemical text mining
- [[lin2024knnbioel]] — kNN-BioEL: Retrieval-enhanced biomedical entity linking
- [[marconato2025reasoning]] — PhD thesis on reasoning shortcuts
- [[steinbeck2020nfdi4chem]] — NFDI4Chem grant proposal
- [[dagdelen2024structured]] — Structured IE from scientific text (Nature Comms)
- [[gupta2022matscibert]] — MatSciBERT domain language model

Updated concept pages: llm-guided-catalyst-discovery, knowledge-graphs-for-catalysis, text-mining-for-catalysis, catalysis-data-infrastructure, chemical-llms
Updated entity pages: catalysis-benchmarks-and-datasets, nfdi4cat, nfdi4chem

## Purpose

This wiki is a **persistent, compounding knowledge base** about the intersection of **catalysis research, knowledge graphs/ontologies, FAIR data infrastructure, and large language models**. It is built and maintained entirely by the LLM from the raw source documents in `docs/literature/`.

## Architecture

### Three Layers

1. **Raw sources** — `docs/literature/*.pdf` — Immutable. The LLM reads but never modifies these.
2. **The wiki** — `docs/llm_wiki/` — LLM-generated markdown files. The LLM owns this entirely.
3. **The schema** — this file (`AGENTS.md`) — Defines conventions and workflows.

### Directory Structure

```
docs/llm_wiki/
├── AGENTS.md            # This file — schema & operating instructions
├── index.md             # Content-oriented catalog of all wiki pages
├── log.md               # Chronological append-only log of operations
├── overview.md          # High-level synthesis & evolving thesis
├── sources/             # One page per ingested source (summary + metadata)
├── concepts/             # Concept pages (cross-cutting topics spanning multiple sources)
├── entities/             # Entity pages (tools, ontologies, organizations, datasets, methods)
└── comparisons/          # Comparison & analysis pages (filed from queries)
```

## Conventions

### File naming
- Source pages: `sources/<short-descriptive-slug>.md` (e.g., `bo-icl-bayesian-optimization-catalysis.md`)
- Concept pages: `concepts/<concept-name>.md` (e.g., `llm-guided-catalyst-discovery.md`)
- Entity pages: `entities/<entity-name>.md` (e.g., `nfdi4cat.md`)
- Comparison pages: `comparisons/<topic>.md`

### Frontmatter (YAML)
Every page starts with YAML frontmatter:
```yaml
---
type: source | concept | entity | comparison
title: "Human-Readable Title"
created: 2025-06-23
updated: 2025-06-23
sources: [source-slug-1, source-slug-2]  # which sources inform this page
tags: [catalysis, llm, ontology, ...]
---
```

### Cross-references
Use Obsidian-style wiki links: `[[page-slug|Display Text]]`

### Citations
When referencing a source, use: `(see [[source-slug]])` or inline: `Source et al. ([[source-slug]])`

## Workflows

### Ingest
1. Read the source PDF (use `pdftotext` to extract text)
2. Create a source page in `sources/` with: title, authors, year, venue, abstract, key findings, methodology, relevance to wiki themes
3. Update relevant concept pages (create if they don't exist)
4. Update relevant entity pages (create if they don't exist)
5. Update `index.md` with the new page
6. Append an entry to `log.md`

### Query
1. Read `index.md` to find relevant pages
2. Read the relevant pages
3. Synthesize an answer with citations
4. If the answer is substantial, file it as a new page (comparison or concept)

### Lint
1. Check for orphan pages (no inbound links)
2. Check for contradictions between pages
3. Check for stale claims superseded by newer sources
4. Suggest new questions to investigate
5. Suggest new sources to acquire

## Domain Themes

This wiki covers several intersecting themes:

1. **LLM-Guided Catalyst Discovery** — Using LLMs for catalyst design, optimization, and reasoning
2. **Knowledge Extraction from Catalysis Literature** — NLP/LLM-based extraction of structured data from papers
3. **Knowledge Graphs for Catalysis** — Building and using KGs in catalysis
4. **Ontologies for Catalysis** — Ontology development, matching, and application
5. **FAIR Data Infrastructure** — Research data management, repositories, standards
6. **Chemical LLMs** — Domain-specialized LLMs for chemistry
7. **Ontology Fundamentals** — Formal ontology theory and methodology
8. **Automated Scientific Discovery** — Closed-loop optimization, autonomous labs, AI-driven experimentation

## Tags Vocabulary
- `catalysis` — Catalysis science and research
- `llm` — Large language models
- `knowledge-graph` — Knowledge graphs
- `ontology` — Ontologies (development, matching, application)
- `fair-data` — FAIR principles and research data management
- `data-infrastructure` — Repositories, platforms, infrastructure
- `bayesian-optimization` — Bayesian optimization methods
- `text-mining` — Information/text extraction from literature
- `materials-discovery` — AI-driven materials discovery
- `chemical-reasoning` — Chemical reasoning with LLMs
- `nfdi` — German NFDI initiative and consortia
- `agent` — LLM agent systems
- `benchmark` — Benchmarks and datasets
- `ontology-matching` — Ontology alignment/matching
- `formal-ontology` — Formal ontology theory
- `protein-engineering` — Protein/enzyme engineering
