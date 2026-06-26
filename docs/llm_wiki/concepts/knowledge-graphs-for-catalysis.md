---
type: concept
title: "Knowledge Graphs for Catalysis"
created: 2025-06-23
updated: 2026-06-26
sources: [kg-survey-ai-for-science, kg-hub, llm-kg-ontology-generation, kg-heterogeneous-catalysis-review, knowledge-extraction-catalysis-literature, spires-ontogpt, zhang2024edc, dagdelen2024structured, swain2016chemdataextractor, gupta2022matscibert, llm-kg-extraction-tables-materials, retrieval-augmented-generation-knowledge-intensive-nlp, rag-survey, self-rag]
tags: [knowledge-graph, catalysis, ontology, llm, text-mining]
---

# Knowledge Graphs for Catalysis

## Overview
Knowledge graphs (KGs) provide a structured, machine-readable format for organizing catalysis knowledge — materials, reactions, conditions, mechanisms — that is dispersed across literature. They serve as the semantic backbone for FAIR data and as knowledge infrastructure for AI-guided discovery.

## Construction Methods

### Ontology-Guided Text Mining
- [[knowledge-extraction-catalysis-literature]]: First benchmark dataset for catalysis IE. Six entity types (catalyst, reaction, reactant, product, characterization, treatment). 90% extraction accuracy.
- [[catminer-llm-catalysis-extraction]]: CatMiner uses LLMs for structure-environment-property extraction. LLM-agnostic (GPT, Llama, DeepSeek).
- [[spires-ontogpt]]: SPIRES uses zero-shot LLM prompting to populate arbitrary knowledge schemas with ontology grounding.

### LLM-Based KG/Ontology Generation
- [[llm-kg-ontology-generation]]: Zero-shot, end-to-end ontology + KG generation from scientific literature using open-source LLMs. First KG for Single Atom Catalysis.
- [[llm-kg-extraction-tables-materials]]: Semi-automated pipeline that turns non-standardized materials-science tables into KG structures using LLM extraction plus rule-based feedback.

### RAG and Evidence Grounding
- [[retrieval-augmented-generation-knowledge-intensive-nlp]]: Establishes parametric-plus-non-parametric memory for grounded generation.
- [[rag-survey]]: Maps naive, advanced, and modular RAG, including evaluation concerns.
- [[self-rag]]: Adds adaptive retrieval and critique, useful for judging whether KG or source evidence is needed before generation.

### Infrastructure
- [[kg-hub]]: KG-Hub provides modular ETL for KG construction with Biolink Model compliance. Graph ML tools integrated.
- [[kg-heterogeneous-catalysis-review]]: Reviews catalysis KGs — from thousands to >1M triples. Covers construction, population, maintenance.

## Applications
- **Catalyst recommendation**: KG + LLM-driven RAG for natural-language queries
- **Reaction mechanism discovery**: Linking entities across data sources
- **Data integration**: FAIR data management across heterogeneous sources
- **RAG enhancement**: KG + ontology improves RAG quality vs KG-only (see [[llm-kg-ontology-generation]])

## Challenges
- Data heterogeneity across catalysis subdomains
- Ontology alignment/matching (see [[ontology-matching]])
- Long-term graph curation and maintenance
- Community standards for ontology development
- Embedding structured knowledge into LLM workflows
- Faithful retrieval from KGs and source documents, especially in long-context settings

## Vision
[[kg-survey-ai-for-science]] envisions self-updating SciKGs co-evolving with LLMs, embodied within AI scientists as core infrastructure for autonomous scientific discovery.
