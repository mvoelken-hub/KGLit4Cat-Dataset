---
type: source
title: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
created: 2026-06-26
updated: 2026-06-26
sources: [sentence-bert]
tags: [llm, embeddings, information-retrieval]
---

# Sentence-BERT

**Authors:** Nils Reimers and Iryna Gurevych
**Year:** 2019
**Venue:** EMNLP-IJCNLP
**File:** `docs/literature/D19-1410.pdf`

## Abstract
Adapts BERT with siamese and triplet network structures to produce sentence embeddings that can be compared efficiently with cosine similarity. This makes semantic search, clustering, and large-scale similarity retrieval practical.

## Key Findings
- Replaces expensive cross-encoder pair scoring with reusable sentence vectors.
- Reduces a 10,000-sentence similarity search from many hours to seconds.
- Maintains strong semantic textual similarity performance.
- Underlies many modern embedding-based retrieval pipelines.

## Relevance
Directly relevant to semantic source retrieval, ontology candidate search, and evidence-context ranking in LLM extraction systems. Relevant to [[retrieval-augmented-generation]] and [[ontology-matching]].
