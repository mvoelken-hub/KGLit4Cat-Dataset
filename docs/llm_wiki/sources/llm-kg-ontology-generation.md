---
type: source
title: "Scientific Knowledge Graph and Ontology Generation Using Open Large Language Models"
created: 2025-06-23
updated: 2025-06-23
sources: [llm-kg-ontology-generation]
tags: [knowledge-graph, ontology, llm, catalysis]
---

# LLM-Based KG & Ontology Generation (Oarga et al.)

**Authors:** Alexandru Oarga, Matthew Hart, Andres M. Bran, Magdalena Lederbauer, Philippe Schwaller
**Year:** 2026
**Venue:** Digital Discovery (RSC)
**File:** `docs/literature/367d68117ef1abf15806cf2e155e4249ac77689efbc0c8b326cea7cbe06a468c.pdf`

## Abstract
Proposes a novel method using LLMs for zero-shot, end-to-end ontology and KG generation from scientific literature, using exclusively open-source LLMs. Evaluated on reconstructing an existing KG of chemical elements/functional groups. Applied to Single Atom Catalysts (SACs), creating the first domain-specific ontology and KG for SAC.

## Key Findings
- Zero-shot pipeline for vocabulary and taxonomy extraction from scientific literature
- Outperforms existing methods in KG reconstruction (chemical elements/functional groups)
- First domain-specific ontological schema and KG for Single Atom Catalysis
- Combination of extracted knowledge schema + KG improves RAG quality (vs KG-only GraphRAG)
- Open-source LLMs only — no fine-tuning required

## Methodology
- Ontological knowledge schema + minimal manual intervention
- LLM-based vocabulary/taxonomy extraction
- KG population from extracted triples
- Evaluation: reconstruct existing KG, then apply to novel domain (SAC)

## Relevance
Directly connects [[knowledge-graphs-for-catalysis]], [[ontology-development]], and [[llm-for-catalysis-data-extraction]]. Demonstrates LLMs can automate ontology construction.