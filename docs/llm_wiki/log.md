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