---
type: source
title: "MatSciBERT: A Materials Domain Language Model for Text Mining and Information Extraction"
created: 2025-06-23
updated: 2025-06-23
sources: [gupta2022matscibert]
tags: [llm, text-mining, materials-discovery]
---

# MatSciBERT

**Authors:** Tanishq Gupta, Mohd Zaki, N. M. Anoop Krishnan, Mausam
**Year:** 2022
**Venue:** npj Comput. Mater. 8(1): 102
**File:** `docs/literature/s41524-022-00784-w.pdf`

## Abstract
Materials-aware BERT model trained on large corpus of peer-reviewed materials science publications. Outperforms SciBERT on three downstream tasks: named entity recognition, relation classification, and abstract classification. Pre-trained weights publicly available.

## Key Findings
- Domain-specific pre-training on materials science corpus
- Outperforms SciBERT (trained on general science corpus)
- SOTA on NER, relation classification, abstract classification
- Pre-trained weights publicly accessible
- Used in combination with ChemDataExtractor for database creation

## Relevance
Domain-specific language model for materials science text. Precursor to LLM-based approaches. Relevant to [[chemical-llms]] and [[text-mining-for-catalysis]].