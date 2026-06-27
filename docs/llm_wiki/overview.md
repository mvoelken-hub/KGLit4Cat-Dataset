---
type: concept
title: "Overview: Catalysis, LLMs, Knowledge Graphs, and FAIR Data"
created: 2025-06-23
updated: 2026-06-26
sources: [all]
tags: [catalysis, llm, knowledge-graph, ontology, fair-data]
---

# Overview: The Convergence of Catalysis, LLMs, Knowledge Graphs, and FAIR Data

## Evolving Thesis

The literature in this collection reveals a rapidly converging landscape where four previously distinct domains are intertwining:

1. **Catalysis science** — the multi-disciplinary study of catalysts (heterogeneous, homogeneous, electro-, photo-, bio-catalysis)
2. **Large language models** — general-purpose and domain-specialized AI for chemistry
3. **Knowledge graphs & ontologies** — structured knowledge representation for machine-readable catalysis data
4. **FAIR data infrastructure** — research data management following Findable, Accessible, Interoperable, Reusable principles

The central thesis emerging from this literature: **AI-guided catalysis discovery requires all four pillars working together.** LLMs without structured knowledge hallucinate. Knowledge graphs without LLMs can't ingest literature at scale. FAIR data without ontologies isn't interoperable. And none of it matters if the community doesn't share data.

## The State of the Field (2022-2026)

### The Data Problem
Catalysis has a dataset-to-article ratio of ~1/100 — data is simply not shared ([[data-key-resource-catalysis]]). Even when shared, it lacks standardization and metadata, making it unusable for machine learning. This is the fundamental bottleneck. Germany's NFDI initiative (NFDI4Cat, NFDI4Chem) is the most concerted effort to fix this, building repositories ([[repo4cat]]), vocabularies ([[voc4cat]]), metadata profiles ([[chemdcat-ap]]), and workflows ([[mardiflow]]).

Adjacent FAIR implementation literature sharpens what "FAIR" operationally means: practical recipes and maturity guidance ([[fair-cookbook]]), database schemas/ETL/APIs for chemical laboratories ([[chemical-data-storage-architectures]]), governance and stewardship in industrial R&D ([[fair-biopharma-rd]]), access restrictions and Semantic Web grounding for digital twins ([[fair-digital-twins]]), and FAIRmat/NOMAD-style federated materials infrastructure ([[fairmat-materials-research]]).

### The LLM Revolution in Catalysis
LLMs have entered catalysis research through multiple pathways:
- **As optimizers**: BO-ICL ([[bo-icl-bayesian-optimization-catalysis]]) and GOLLuM ([[gollum-uncertainty-calibrated-llm]]) use LLMs as surrogate models in Bayesian optimization, eliminating the need for domain-specific feature engineering. GOLLuM's key insight is that uncertainty calibration — not raw capability — is what makes LLMs useful for real experimental campaigns.
- **As reasoning agents**: ChemReasoner ([[monte-carlo-thought-search]]) uses MCTS over prompt space for combinatorial catalyst search. Adsorb-Agent ([[adsorb-agent]]) navigates adsorption configuration spaces.
- **As domain specialists**: CataLM ([[catalm-catalyst-design-llm]]) and ChemDFM-R ([[chemdfm-r]]) are fine-tuned for catalysis/chemistry, incorporating atomized chemical knowledge.
- **As text miners**: CatMiner ([[catminer-llm-catalysis-extraction]]) and SPIRES ([[spires-ontogpt]]) extract structured data from literature, populating databases and knowledge graphs.

### Knowledge Graphs as the Bridge
KGs sit between raw literature and AI applications. [[knowledge-graphs-for-catalysis]] reviews the landscape: from ontology-guided text mining to LLM-driven graph generation ([[llm-kg-ontology-generation]]) to RAG-enhanced natural language queries. The vision is self-updating SciKGs co-evolving with LLMs ([[kg-survey-ai-for-science]]).

### LLM Foundations and Reliability
The fresh LLM foundations batch clarifies the technical substrate behind these applied systems. [[attention-is-all-you-need]] supplies the Transformer architecture; [[bert]], [[scibert]], and [[sentence-bert]] explain the encoder and embedding lineage for scientific NLP and retrieval; [[gpt3-few-shot-learners]] and [[instructgpt-rlhf]] explain prompt-based and instruction-following LLM use. The main operational lesson is that extraction quality depends on more than model capability: retrieval design ([[retrieval-augmented-generation]], [[self-rag]]), context placement ([[lost-in-the-middle]]), decoding choices ([[neural-text-degeneration]]), and structured-output controls ([[grammar-constrained-decoding]], [[structured-output-ie-go]]) all shape whether outputs are faithful and schema-valid.

### Ontology Development
Ontology work ranges from foundational theory ([[guarino-fois98]], [[bfo-textbook]]) to catalysis-specific implementations ([[chemdcat-ap]], [[ontologies4cat]], [[ontology-reaction-classification]]). LLMs are now being used to automate ontology construction ([[llm-kg-ontology-generation]]) and matching ([[complex-ontology-matching-llm]], [[lakermap-ontology-matching]], [[hgnn-ontology-matching]]).

## Key Tensions and Open Questions

1. **Fine-tuning vs. prompting**: Should we build domain-specific LLMs (CataLM, ChemDFM-R) or use general LLMs with clever prompting (BO-ICL, GOLLuM)? Both show promise; the answer may depend on the task.

2. **Automation vs. human oversight**: Autonomous discovery is the goal, but all successful systems still involve humans. Where should the human be in the loop?

3. **Standardization vs. flexibility**: FAIR principles demand standards, but catalysis is heterogeneous across subdomains. ChemDCAT-AP's modular approach (domain-agnostic core + domain-specific extensions) is one solution.

4. **Open vs. proprietary**: The community strongly favors open-source (NFDI, SPIRES, CatMiner supports open models), but best LLM performance often comes from proprietary models (GPT-4o in Adsorb-Agent).

5. **Data scarcity vs. data quality**: The community needs more data, but also better data. Mandatory FAIR depositing before publication ([[data-key-resource-catalysis]]) addresses both.

6. **Long context vs. curated evidence**: More tokens do not guarantee better extraction. [[lost-in-the-middle]] suggests that SIMONE-like workflows should retrieve and order compact evidence windows rather than dumping full documents into prompts.

7. **Free-form generation vs. schema validity**: [[grammar-constrained-decoding]] and [[structured-output-ie-go]] strengthen the case for separating evidence interpretation from schema serialization.

## Map of the Literature

### Core LLM + Catalysis Papers
- [[bo-icl-bayesian-optimization-catalysis]] — BO-ICL (2023)
- [[gollum-uncertainty-calibrated-llm]] — GOLLuM (2025)
- [[monte-carlo-thought-search]] — ChemReasoner (2023)
- [[adsorb-agent]] — Adsorb-Agent (2024)
- [[catalm-catalyst-design-llm]] — CataLM (2024)
- [[chemdfm-r]] — ChemDFM-R (2025)
- [[catminer-llm-catalysis-extraction]] — CatMiner (2025)
- [[knowledge-extraction-catalysis-literature]] — Catalysis IE benchmark (2022)
- [[alchemybench]] — AlchemyBench (2025)
- [[tos-catalyst-reactivity-ai]] — AI for TOS modeling (2025)

### Knowledge Graphs & Ontologies
- [[kg-survey-ai-for-science]] — KG survey (2026)
- [[kg-heterogeneous-catalysis-review]] — KG in hetero. catalysis review (2025)
- [[llm-kg-ontology-generation]] — LLM KG generation (2026)
- [[kg-hub]] — KG-Hub (2023)
- [[spires-ontogpt]] — SPIRES (2024)
- [[ols4]] — OLS4 (2025)
- [[ontologies4cat]] — Ontologies4Cat (2024)
- [[chemdcat-ap]] — ChemDCAT-AP (2026)
- [[ontology-reaction-classification]] — Reac4Cat extension (2025)
- [[complex-ontology-matching-llm]] — LLM ontology matching (2025)
- [[lakermap-ontology-matching]] — LaKERMap (2023)
- [[hgnn-ontology-matching]] — HGNN OM (2026)

### FAIR Data Infrastructure
- [[nfdi4cat-unified-infrastructure]] — NFDI4Cat concept (2021)
- [[data-key-resource-catalysis]] — Community account (2025)
- [[nfdi4cat-whitepaper]] — White paper (2024)
- [[repo4cat]] — Repo4Cat (2025)
- [[nfdi4chem]] — NFDI4Chem (2023)
- [[nomad-catalysis-plugin]] — NOMAD plugin (2025)
- [[orchestrating-catalysis-data]] — FAIR guide
- [[mardiflow]] — MaRDIFlow (2024)
- [[mardiflow-voc4cat]] — MaRDIFlow + Voc4Cat (2025)

- [[fair-cookbook]] - FAIR implementation recipes (2023)
- [[chemical-data-storage-architectures]] - chemistry databases and FAIR data (2022)
- [[fair-biopharma-rd]] - industrial FAIR implementation (2019)
- [[fair-digital-twins]] - FAIR and Semantic Web for digital twins (2023)
- [[fairmat-materials-research]] - FAIRmat/NOMAD materials infrastructure (2022)

### Foundational / Background
- [[llm-foundations]] - LLM architecture, tokenization, instruction tuning, serving
- [[retrieval-augmented-generation]] - retrieval, source grounding, long-context limits
- [[structured-output-reliability]] - constrained decoding and schema-valid extraction
- [[guarino-fois98]] — Formal ontology (1998)
- [[bfo-textbook]] — BFO textbook
- [[ontocape]] — OntoCAPE (2010)
- [[heterogeneous-catalysis-history]] — Catalysis history (2012)
- [[low-energy-catalysis-review]] — Low-energy catalysis (2024)
- [[catalyst-informatics]] — Catalyst informatics (2023)
- [[autonomous-protein-engineering]] — Autonomous protein eng. (2025)

### Peripheral
- [[eclass-semantic-search]], [[ontology-energy-systems]], [[llama3-lora-qlora]], [[epicure-food-embeddings]], [[methane-reforming-reporting-framework]], textbooks
