---
type: concept
title: "Text Mining and Knowledge Extraction from Catalysis Literature"
created: 2025-06-23
updated: 2026-06-29
sources: [knowledge-extraction-catalysis-literature, catminer-llm-catalysis-extraction, spires-ontogpt, llm-kg-ontology-generation, kg-heterogeneous-catalysis-review, dagdelen2024structured, schilling2025text, zhang2024chemicaltextmining, openchemie, llm-generative-information-extraction-survey, swain2016chemdataextractor, gupta2022matscibert, scibert, grammar-constrained-decoding, structured-output-ie-go, lost-in-the-middle, llm-kg-extraction-tables-materials]
tags: [text-mining, catalysis, llm, knowledge-graph]
---

# Text Mining and Knowledge Extraction from Catalysis Literature

## Overview
Most catalysis knowledge is locked in unstructured scientific literature. Extracting structured data — entities, relationships, properties, conditions — is essential for building databases, knowledge graphs, and enabling AI-driven discovery.

## Approaches

### Traditional NLP Pipeline
[[knowledge-extraction-catalysis-literature]]: First benchmark dataset for catalysis IE. Pipeline:
1. Article selection (binary classifier for relevance)
2. Named entity recognition (6 types: catalyst, reaction, reactant, product, characterization, treatment)
3. 90% extraction accuracy
4. Entity-aware search engine + correlation analysis

### LLM-Based Text Mining
[[catminer-llm-catalysis-extraction]] (CatMiner):
- LLM-agnostic (GPT, Llama, DeepSeek without modification)
- User-specified extraction schema
- Key strategies: domain knowledge injection, iterative prompting, document-wide context
- Case study: oxidative coupling of methane

### Zero-Shot Schema Population
[[spires-ontogpt]] (SPIRES):
- Recursive LLM prompt interrogation for arbitrary knowledge schemas
- Ontology-based grounding (bypasses hallucination)
- Handles nested complex classes
- No training data needed

### End-to-End KG + Ontology Generation
[[llm-kg-ontology-generation]]:
- Zero-shot pipeline: vocabulary → taxonomy → KG
- Open-source LLMs only
- First KG + ontology for Single Atom Catalysis

### Structured Output and Evidence Reliability
- [[grammar-constrained-decoding]]: uses formal grammars to guarantee valid output structures for IE, entity disambiguation, and parsing without fine-tuning.
- [[structured-output-ie-go]]: separates content generation from formatting via Generate-and-Organize prompting, improving zero-shot NER and relation extraction.
- [[llm-generative-information-extraction-survey]]: surveys LLM-based generative IE across NER, relation extraction, event extraction, universal IE, low-resource techniques, retrieval, and self-improvement.
- [[lost-in-the-middle]]: warns that long-document extraction can fail when relevant evidence is buried in the middle of large contexts.
- [[llm-kg-extraction-tables-materials]]: extends extraction beyond prose to non-standardized R&D tables, producing graph structures from tabular materials data.
- [[openchemie]]: shows that chemistry reaction extraction often requires integrating text, tables, and figures, including R-group resolution and reaction-condition alignment.

## Key Challenges
- Most catalysis data includes environmental/operating parameters that standard materials IE doesn't capture
- Reporting standards vary widely across papers
- LLM hallucination and inconsistency in extraction
- Entity grounding to persistent identifiers
- Scaling to hundreds of thousands of papers
- Schema-valid serialization and source-grounded evidence selection
- Table and spreadsheet extraction, not only article prose
- Multimodal evidence alignment across figures, tables, diagrams, OCR, and prose

## Synergy with Knowledge Graphs
Text mining feeds [[knowledge-graphs-for-catalysis]]. Extracted triples populate KGs, which in turn enable RAG-based natural language queries (see [[kg-heterogeneous-catalysis-review]]).
