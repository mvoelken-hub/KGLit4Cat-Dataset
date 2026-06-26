---
type: concept
title: "Retrieval-Augmented Generation"
created: 2026-06-26
updated: 2026-06-26
sources: [bm25-probabilistic-relevance, sentence-bert, retrieval-augmented-generation-knowledge-intensive-nlp, rag-survey, self-rag, lost-in-the-middle]
tags: [llm, rag, information-retrieval]
---

# Retrieval-Augmented Generation

RAG connects LLM generation to external, inspectable knowledge. In this wiki, it is the bridge between raw scientific documents, ontology resources, knowledge graphs, and source-grounded extraction.

## Retrieval Foundations
- [[bm25-probabilistic-relevance]] provides the lexical retrieval baseline: exact terms, identifiers, and domain vocabulary still matter.
- [[sentence-bert]] provides efficient semantic embeddings for similarity search and clustering.
- Hybrid retrieval is therefore important for catalysis: lexical search catches precise symbols and material names, while embeddings catch paraphrases and conceptual similarity.

## Generative RAG
- [[retrieval-augmented-generation-knowledge-intensive-nlp]] introduces the core parametric-plus-non-parametric memory pattern.
- [[rag-survey]] organizes the field into naive, advanced, and modular RAG and emphasizes evaluation.
- [[self-rag]] adds adaptive retrieval and self-critique, which matters when not every extraction step needs external evidence.

## Long-Context Caution
[[lost-in-the-middle]] shows that more context is not automatically better. If relevant evidence is placed in the middle of a long prompt, models may underuse it. RAG systems must therefore rank, filter, and order evidence deliberately.

## Relevance to SIMONE
SIMONE-like metadata extraction should treat RAG as evidence selection, not just prompt stuffing. The best workflow retrieves small, source-located evidence windows, mixes lexical and semantic search, preserves provenance, and validates output against the retrieved source fragments.
