# Wiki Log

Append-only chronological record of wiki operations.

Format: `## [YYYY-MM-DD] operation | description`

---

## [2025-06-23] init | LLM Wiki created
Initial wiki setup based on Karpathy's LLM Wiki pattern. Schema (`AGENTS.md`), index, and log created.

## [2025-06-23] ingest | Batch ingestion of 48 PDFs (47 sources) from docs/literature
Ingested all 48 PDFs from `docs/literature/` in a single batch. MTSR_2025_paper_1431-1.pdf is the conference version of 2602.01822v1.pdf (both ChemDCAT-AP), so combined into one source page. Created:
- 47 source summary pages (sources/)
- 10 concept pages (concepts/) covering LLM-guided catalyst discovery, Bayesian optimization, KGs, ontologies, FAIR data, text mining, chemical LLMs, automated discovery, formal ontology, ontology matching
- 6 entity pages (entities/) covering NFDI4Cat, NFDI4Chem, Voc4Cat, catalysis ontologies, ontology resources, benchmarks/datasets
- overview.md (high-level synthesis with evolving thesis)
- index.md (content-oriented catalog)
Papers span catalysis + LLMs, knowledge graphs, ontologies, FAIR data infrastructure, and related domains.

## [2025-06-23] ingest | 16 new sources added
Added source pages for 16 newly acquired PDFs. Updated 5 concept pages and 3 entity pages with new cross-references. See AGENTS.md for details.

## [2026-06-26] ingest | LLM foundations, RAG, and structured-output batch
Added 19 source pages from fresh `docs/literature/` PDFs covering Transformer/BERT/SciBERT foundations, tokenization, few-shot and instruction-tuned LLMs, RAG, Self-RAG, long-context behavior, constrained decoding, structured IE prompting, LLM serving, tool use, and table-to-KG extraction. Skipped the duplicate `2025.findings-acl.558 (1).pdf` because [[kim2025angel]] already covers ANGEL.

Created concept pages:
- [[llm-foundations]]
- [[retrieval-augmented-generation]]
- [[structured-output-reliability]]

Updated concept pages: [[chemical-llms]], [[text-mining-for-catalysis]], [[knowledge-graphs-for-catalysis]], [[automated-scientific-discovery]]
Updated catalog: [[index]]

## [2026-06-27] ingest | FAIR implementation and materials data infrastructure batch
Added 5 source pages from newly added `docs/literature/` PDFs:
- [[fair-cookbook]] - practical FAIRification recipes and maturity guidance
- [[chemical-data-storage-architectures]] - database schemas, ETL, APIs, and FAIR chemical lab data
- [[fair-biopharma-rd]] - FAIR implementation, governance, stewardship, and knowledge representation in industrial R&D
- [[fair-digital-twins]] - FAIR principles, Semantic Web technologies, access restrictions, and provenance for digital twins
- [[fairmat-materials-research]] - FAIRmat/NOMAD-style federated materials infrastructure, metadata, ontologies, and AI-ready data

Updated concept pages: [[catalysis-data-infrastructure]]
Updated catalog and synthesis: [[index]], [[overview]]

## [2026-06-27] ingest | Biocatalysis and catalysis data infrastructure batch
Added 4 source pages from newly added `docs/literature/` PDFs:
- [[enzymeml-fair-biocatalysis]] - EnzymeML for FAIR enzymology/biocatalysis data management
- [[fair-biocatalytic-science]] - FAIR data and FAIR software for biocatalytic science
- [[electrocatalysis-research-database]] - Catalysis-Hub Experimental database for electrocatalysis
- [[catalysis-data-infrastructure-ukch]] - UK Catalysis Hub CDI catalogue/prototype

Updated concept page: [[catalysis-data-infrastructure]]
Updated catalog and synthesis: [[index]], [[overview]]
Updated thesis bibliography: `docs/thesis/bibliography/references.bib`

## [2026-06-28] ingest | Reac4Cat-Ontology source added
Added source page for newly added `docs/literature/s13222-024-00476-3.pdf`:
- [[reac4cat-ontology]] - OWL description-logic / GCA-based reaction and catalysis knowledge graph for reaction-role and catalyst inference, demonstrated with EnzymeML and DWSIM-linked process data

Updated concept/entity pages: [[ontology-development]], [[catalysis-ontologies]]
Updated catalog and synthesis: [[index]], [[overview]]
Updated thesis bibliography: `docs/thesis/bibliography/references.bib`
