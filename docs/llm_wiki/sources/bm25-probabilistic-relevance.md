---
type: source
title: "The Probabilistic Relevance Framework: BM25 and Beyond"
created: 2026-06-26
updated: 2026-06-26
sources: [bm25-probabilistic-relevance]
tags: [information-retrieval, rag, foundational]
---

# BM25 and the Probabilistic Relevance Framework

**Authors:** Stephen Robertson and Hugo Zaragoza
**Year:** 2009
**Venue:** Foundations and Trends in Information Retrieval 3(4)
**File:** `docs/literature/1500000019.pdf`

## Abstract
Survey and formalization of probabilistic relevance ranking, including BM25. It explains how term frequency, inverse document frequency, document length normalization, and relevance assumptions lead to practical ranking functions for document retrieval.

## Key Findings
- BM25 is a robust lexical ranking baseline for ad hoc retrieval.
- The probability ranking principle links retrieval ranking to estimated relevance.
- Term weighting and document-length normalization remain useful even in neural retrieval eras.
- Lexical retrieval is often complementary to dense semantic retrieval.

## Relevance
Baseline retrieval theory for RAG and literature search. For SIMONE-like workflows, BM25 remains a strong component for source chunk retrieval, citation traceability, and fallback search when embeddings miss exact terms. Relevant to [[retrieval-augmented-generation]].
