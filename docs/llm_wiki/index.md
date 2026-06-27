# Wiki Index

Content-oriented catalog of all pages in the LLM Wiki. Organized by category.

---

## Overview
- [[overview|Overview: Catalysis, LLMs, KGs, and FAIR Data]] — High-level synthesis and evolving thesis

## Concept Pages

### Catalysis + AI
- [[llm-guided-catalyst-discovery|LLM-Guided Catalyst Discovery]] — LLMs as optimizers, agents, and reasoning engines for catalyst design
- [[bayesian-optimization-for-catalysis|Bayesian Optimization for Catalysis]] — BO with LLM surrogates (BO-ICL, GOLLuM)
- [[automated-scientific-discovery|Automated Scientific Discovery]] — Closed-loop optimization, autonomous labs, AI scientists
- [[chemical-llms|Chemical LLMs]] — Domain-specialized LLMs for chemistry/catalysis
- [[llm-foundations|LLM Foundations]] - Transformer architecture, tokenization, instruction tuning, scientific encoders, and serving

### Knowledge & Data
- [[knowledge-graphs-for-catalysis|Knowledge Graphs for Catalysis]] — KG construction, population, and applications
- [[text-mining-for-catalysis|Text Mining for Catalysis Literature]] — IE from catalysis papers using NLP/LLMs
- [[catalysis-data-infrastructure|FAIR Data Infrastructure for Catalysis]] — NFDI, repositories, standards, workflows
- [[retrieval-augmented-generation|Retrieval-Augmented Generation]] - Evidence retrieval, RAG design, long-context limits, and source grounding
- [[structured-output-reliability|Structured Output Reliability]] - Schema-valid LLM extraction, constrained decoding, and evidence-aware validation

### Ontology
- [[ontology-development|Ontology Development for Catalysis]] — Domain ontologies, vocabularies, application profiles
- [[ontology-matching|Ontology Matching]] — Aligning ontologies using LLMs, GNNs, and self-supervised learning
- [[formal-ontology|Formal Ontology: Theory and Foundations]] — Guarino, BFO, foundational definitions

---

## Entity Pages

### Organizations & Consortia
- [[nfdi4cat|NFDI4Cat]] — German national RDI for catalysis
- [[nfdi4chem|NFDI4Chem]] — German national RDI for chemistry

### Vocabularies & Ontologies
- [[voc4cat|Voc4Cat]] — SKOS vocabulary for catalysis
- [[catalysis-ontologies|Catalysis Ontologies and Vocabularies]] — Survey of domain ontologies

### Infrastructure & Resources
- [[ontology-resources|Ontology Resources and Infrastructure]] — OLS4, KG-Hub, SPIRES, standards
- [[catalysis-benchmarks-and-datasets|Catalysis Benchmarks and Datasets]] — OMG/AlchemyBench, IE benchmarks, BioFuelQR

---

## Source Pages

### LLM + Catalysis (Core)
| Page | Title | Year | Tags |
|------|-------|------|------|
| [[bo-icl-bayesian-optimization-catalysis]] | BO-ICL: Bayesian Optimization of Catalysis with In-Context Learning | 2023 | catalysis, llm, BO |
| [[gollum-uncertainty-calibrated-llm]] | GOLLuM: LLMs as Uncertainty-Calibrated Optimizers | 2025 | catalysis, llm, BO |
| [[monte-carlo-thought-search]] | Monte Carlo Thought Search (ChemReasoner) | 2023 | catalysis, llm, agent |
| [[adsorb-agent]] | Adsorb-Agent: LLM Agent for Adsorption Configurations | 2024 | catalysis, llm, agent |
| [[catalm-catalyst-design-llm]] | CataLM: Empowering Catalyst Design Through LLMs | 2024 | catalysis, llm |
| [[chemdfm-r]] | ChemDFM-R: Chemical Reasoning LLM | 2025 | llm, chemical-reasoning |
| [[catminer-llm-catalysis-extraction]] | CatMiner: LLM Text Mining for Heterogeneous Catalysis | 2025 | catalysis, llm, text-mining |
| [[knowledge-extraction-catalysis-literature]] | Knowledge Extraction from Catalysis Literature | 2022 | catalysis, text-mining |
| [[alchemybench]] | AlchemyBench + Open Materials Guide | 2025 | materials-discovery, llm, benchmark |
| [[tos-catalyst-reactivity-ai]] | Modeling TOS Catalyst Reactivity via AI | 2025 | catalysis, materials-discovery |

### Chemical Text Mining & IE
| Page | Title | Year | Tags |
|------|-------|------|------|
| [[swain2016chemdataextractor]] | ChemDataExtractor | 2016 | text-mining |
| [[gupta2022matscibert]] | MatSciBERT | 2022 | llm, text-mining |
| [[dagdelen2024structured]] | Structured IE from Scientific Text | 2024 | llm, text-mining |
| [[schilling2025text]] | From Text to Insight: LLMs for Chemical Extraction | 2025 | llm, text-mining |
| [[zhang2024chemicaltextmining]] | Fine-Tuning LLMs for Chemical Text Mining | 2024 | llm, text-mining |
| [[zhang2024edc]] | EDC: LLM-based KG Construction | 2024 | knowledge-graph, llm |
| [[kim2025angel]] | ANGEL: Negative Sample Learning for BioEL | 2025 | llm, text-mining |
| [[lin2024knnbioel]] | kNN-BioEL: Retrieval-Enhanced BioEL | 2024 | llm, text-mining |
| [[zi2026shattering]] | ShatterMed-QA: Multi-hop Medical Reasoning | 2026 | llm, knowledge-graph |
| [[marconato2025reasoning]] | Reasoning Shortcuts (PhD Thesis) | 2025 | llm, chemical-reasoning |
| [[scibert]] | SciBERT: Scientific Language Model | 2019 | llm, text-mining |
| [[grammar-constrained-decoding]] | Grammar-Constrained Decoding for Structured NLP | 2023 | llm, structured-output |
| [[structured-output-ie-go]] | Generate-and-Organize for Structured IE Output | 2024 | llm, structured-output |
| [[llm-kg-extraction-tables-materials]] | LLM KG Extraction from Materials Tables | 2025 | knowledge-graph, text-mining |

### LLM Foundations, RAG & Structured Output
| Page | Title | Year | Tags |
|------|-------|------|------|
| [[attention-is-all-you-need]] | Attention Is All You Need | 2017 | llm, transformer |
| [[bm25-probabilistic-relevance]] | BM25 and Probabilistic Relevance | 2009 | information-retrieval, rag |
| [[sentencepiece]] | SentencePiece Tokenizer | 2018 | llm, tokenization |
| [[bert]] | BERT | 2019 | llm, transformer |
| [[sentence-bert]] | Sentence-BERT | 2019 | embeddings, retrieval |
| [[neural-text-degeneration]] | The Curious Case of Neural Text Degeneration | 2020 | llm, decoding |
| [[gpt3-few-shot-learners]] | Language Models Are Few-Shot Learners | 2020 | llm, in-context learning |
| [[retrieval-augmented-generation-knowledge-intensive-nlp]] | Retrieval-Augmented Generation for Knowledge-Intensive NLP | 2020 | llm, rag |
| [[instructgpt-rlhf]] | Training Language Models to Follow Instructions with Human Feedback | 2022 | llm, alignment |
| [[toolformer]] | Toolformer | 2023 | llm, tool-use |
| [[pagedattention-vllm]] | PagedAttention / vLLM | 2023 | llm, serving |
| [[rag-survey]] | RAG Survey | 2024 | rag, survey |
| [[self-rag]] | Self-RAG | 2024 | rag, self-reflection |
| [[lost-in-the-middle]] | Lost in the Middle | 2024 | llm, long-context |
| [[llm-survey-zhao2026]] | A Survey of Large Language Models | 2026 | llm, survey |

### Knowledge Graphs & Ontologies
| Page | Title | Year | Tags |
|------|-------|------|------|
| [[kg-survey-ai-for-science]] | KGs in AI for Science — Survey | 2026 | knowledge-graph, llm |
| [[kg-heterogeneous-catalysis-review]] | KGs in Heterogeneous Catalysis — Review | 2025 | knowledge-graph, catalysis |
| [[llm-kg-ontology-generation]] | LLM-Based KG & Ontology Generation | 2026 | knowledge-graph, ontology, llm |
| [[kg-hub]] | KG-Hub: Building Biological KGs | 2023 | knowledge-graph, ontology |
| [[spires-ontogpt]] | SPIRES: Zero-Shot KB Population | 2024 | llm, text-mining, ontology |
| [[ols4]] | OLS4: Ontology Lookup Service | 2025 | ontology, data-infrastructure |
| [[complex-ontology-matching-llm]] | Complex Ontology Matching with LLM Embeddings | 2025 | ontology, ontology-matching, llm |
| [[lakermap-ontology-matching]] | LaKERMap: Self-supervised OM | 2023 | ontology, ontology-matching |
| [[hgnn-ontology-matching]] | Ontology Matching with HGNN | 2026 | ontology, ontology-matching |

### FAIR Data Infrastructure
| Page | Title | Year | Tags |
|------|-------|------|------|
| [[nfdi4cat-unified-infrastructure]] | NFDI4Cat: Unified RDI for Catalysis | 2021 | fair-data, nfdi |
| [[data-key-resource-catalysis]] | Data as a Key Resource in Catalysis | 2025 | fair-data, catalysis |
| [[nfdi4cat-whitepaper]] | NFDI4Cat White Paper: Ontology-based Data Management | 2024 | ontology, nfdi |
| [[repo4cat]] | Repo4Cat: Data Repository for Catalysis | 2025 | data-infrastructure, nfdi |
| [[nfdi4chem]] | NFDI4Chem: Research Data Network for Chemistry | 2023 | fair-data, nfdi |
| [[nomad-catalysis-plugin]] | NOMAD Catalysis Plugin | 2025 | fair-data, catalysis |
| [[orchestrating-catalysis-data]] | Orchestrating Catalysis Data for FAIR | N/A | fair-data, nfdi |
| [[marshall2023digital]] | Achieving Digital Catalysis | 2023 | catalysis, fair-data |
| [[mendes2021opendata]] | Open Data in Catalysis: Big to Small Data | 2021 | catalysis, fair-data |
| [[moshantaf2024advancing]] | FAIR Data in Local Catalysis Infrastructure | 2024 | fair-data, catalysis |
| [[huskova2025improvement]] | Data/Metadata Quality in Catalysis | 2025 | fair-data, nfdi |
| [[cattesthub]] | CatTestHub Benchmarking Database | 2025 | catalysis, benchmark |
| [[steinbeck2020nfdi4chem]] | NFDI4Chem Grant Proposal | 2020 | fair-data, nfdi |
| [[chemdcat-ap]] | ChemDCAT-AP: Semantic Interoperability Profile | 2026 | ontology, fair-data, nfdi |
| [[ontologies4cat]] | Ontologies4Cat: Ontology Landscape for Catalysis | 2024 | ontology, catalysis, nfdi |
| [[ontology-reaction-classification]] | Ontology-Based Reaction Classification Pipeline | 2025 | ontology, catalysis, nfdi |
| [[mardiflow]] | MaRDIFlow: CSE Workflow Framework | 2024 | data-infrastructure, fair-data |
| [[mardiflow-voc4cat]] | MaRDIFlow + Voc4Cat Integration | 2025 | ontology, data-infrastructure, nfdi |
| [[methane-reforming-reporting-framework]] | Standardized Reporting for Methane Reforming | 2026 | catalysis, fair-data |
| [[fair-cookbook]] | FAIR Cookbook: Practical FAIRification Recipes | 2023 | fair-data, training |
| [[chemical-data-storage-architectures]] | Data Storage Architectures for Chemical Discovery | 2022 | chemistry, fair-data, databases |
| [[fair-biopharma-rd]] | FAIR Data Principles in Biopharmaceutical R&D | 2019 | fair-data, industry |
| [[fair-digital-twins]] | FAIR for Digital Twins | 2023 | fair-data, semantic-web, industry |
| [[fairmat-materials-research]] | FAIR Data for Materials Research / FAIRmat | 2022 | fair-data, materials-science, nfdi |
| [[enzymeml-fair-biocatalysis]] | EnzymeML as FAIR Data Management in Biocatalysis | 2021 | biocatalysis, fair-data |
| [[fair-biocatalytic-science]] | FAIR Data and Software in Biocatalytic Science | 2024 | biocatalysis, fair-data, software |
| [[electrocatalysis-research-database]] | Experimental Electrocatalysis Research Database | 2025 | catalysis, fair-data, database |
| [[catalysis-data-infrastructure-ukch]] | UK Catalysis Data Infrastructure | 2022 | catalysis, fair-data, data-infrastructure |

### Foundational & Background
| Page | Title | Year | Tags |
|------|-------|------|------|
| [[guarino-fois98]] | Formal Ontology and Information Systems | 1998 | formal-ontology |
| [[bfo-textbook]] | Building Ontologies with BFO | N/A | formal-ontology |
| [[ontocape]] | OntoCAPE: Ontology for Chemical Process Engineering | 2010 | ontology, catalysis |
| [[catalyst-informatics]] | Catalyst Informatics: Data-Driven Design | 2023 | catalysis, data-infrastructure |
| [[heterogeneous-catalysis-history]] | Past, Present, Future of Heterogeneous Catalysis | 2012 | catalysis |
| [[low-energy-catalysis-review]] | Low-Energy Catalytic Processes Review | 2024 | catalysis |
| [[autonomous-protein-engineering]] | Autonomous Protein Engineering | 2025 | materials-discovery |

### Peripheral
| Page | Title | Year | Tags |
|------|-------|------|------|
| [[eclass-semantic-search]] | ECLASS Semantic Product Search | 2026 | ontology |
| [[ontology-energy-systems]] | Ontology-Driven Dataspaces for Energy Systems | 2026 | ontology, fair-data |
| [[llama3-lora-qlora]] | LLaMA3 LoRA/QLoRA Fine-tuning | 2025 | llm |
| [[epicure-food-embeddings]] | Epicure: Food Ingredient Embeddings | 2026 | — |
| [[digital-humanities-textbook]] | Digital Humanities (textbook) | N/A | — |
| [[informatik-im-fokus]] | Informatik im Fokus (textbook) | N/A | — |
| [[physikalische-chemie-textbook]] | Physikalische Chemie (textbook) | N/A | catalysis |
| [[ifac-optimization]] | IFAC Optimization Paper | 2021 | catalysis |

---

## Tag Index
- **catalysis**: 22 pages
- **llm**: 34 pages
- **ontology**: 16 pages
- **fair-data**: 21 pages
- **knowledge-graph**: 8 pages
- **data-infrastructure**: 13 pages
- **nfdi**: 10 pages
- **bayesian-optimization**: 3 pages
- **text-mining**: 8 pages
- **materials-discovery**: 6 pages
- **ontology-matching**: 4 pages
- **formal-ontology**: 2 pages
- **agent**: 3 pages
- **benchmark**: 2 pages
- **chemical-reasoning**: 2 pages
- **rag**: 5 pages
- **structured-output**: 3 pages
- **biocatalysis**: 2 pages
- **software**: 1 page
- **database**: 1 page
