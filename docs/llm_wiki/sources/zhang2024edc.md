---
type: source
title: "EDC: Extract, Define, Canonicalize — An LLM-based Framework for Knowledge Graph Construction"
created: 2025-06-23
updated: 2025-06-23
sources: [zhang2024edc]
tags: [knowledge-graph, llm, text-mining]
---

# EDC: Extract-Define-Canonicalize

**Authors:** Bowen Zhang, Harold Soh (National University of Singapore)
**Year:** 2024
**Venue:** EMNLP 2024 (pp. 9820–9836)
**File:** `docs/literature/2024.emnlp-main.548.pdf`
**Duplicate/Preprint File:** `docs/literature/2404.03868v2.pdf`

## Abstract
Three-phase framework for knowledge graph construction from text using LLMs: (1) open information extraction, (2) schema definition, (3) post-hoc canonicalization. Flexible: works with or without a pre-defined target schema. Introduces a trained retrieval component to improve extraction via RAG-like schema retrieval.

## Key Findings
- Addresses context window limits: schema doesn't need to fit in prompt
- Auto-constructs schema when none is available, with self-canonicalization
- Trained retrieval component retrieves relevant schema elements per text
- Demonstrated on three KGC benchmarks — extracts high-quality triplets without parameter tuning
- Handles significantly larger schemas than prior work

## Relevance
Key method for LLM-based KG construction. Relevant to [[knowledge-graphs-for-catalysis]] and [[text-mining-for-catalysis]]. Code: https://github.com/clear-nus/edc
